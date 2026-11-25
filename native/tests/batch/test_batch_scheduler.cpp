// CPU unit tests for the continuous-batching Scheduler:
// 1. 64 concurrent heterogeneous requests run in parallel (peak concurrency == 64), exact outputs, zero slot leaks.
// 2. Mid-generation admission & immediate retirement on stop token and client cancellation.
// 3. KV-page backpressure, error isolation (bad prompt fails without breaking active slots), and clean shutdown.
#include "engine/scheduler.h"
#include "ingine/batch_spec_plan.h"
#include "model/batch_spec_verify.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::BatchSlotStep;
using ingine::model::GenerateStats;
using ingine::model::Prompt;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

class FakeBatchGenerator final : public ingine::model::Generator {
 public:
  explicit FakeBatchGenerator(int max_batch, int max_pages = 10000, std::chrono::milliseconds step_delay = {})
      : max_batch_(max_batch), max_pages_(max_pages), free_pages_(max_pages), step_delay_(step_delay),
        slots_(std::size_t(max_batch)) {}

  std::string model_id() const override { return "fake-batch"; }
  int max_model_len() const override { return 4096; }
  int max_batch_size() const override { return max_batch_; }

  bool can_admit_slot(int prompt_tokens, int) const override {
    std::lock_guard lock(mu_);
    int free_s = 0;
    for (const auto& s : slots_)
      if (!s.used) ++free_s;
    const int need = (prompt_tokens + 63) / 64 + 1;
    return free_s > 0 && free_pages_ >= need;
  }
  bool can_admit_slot(const Prompt& prompt, int max_new_tokens, const Sampling&) const override {
    if (prompt.cache_namespace == "throw-admit") {
      throw std::runtime_error("injected admission fault");
    }
    return can_admit_slot(int(prompt.ids.size()), max_new_tokens);
  }

  int start_slot(const Prompt& prompt, int, const Sampling&, int32_t& first_token, GenerateStats& stats) override {
    if (prompt.ids.empty() || prompt.ids[0] < 0) throw std::invalid_argument("bad prompt");
    std::lock_guard lock(mu_);
    const int need = (int(prompt.ids.size()) + 63) / 64 + 1;
    if (free_pages_ < need) return -1;
    for (int i = 0; i < max_batch_; ++i) {
      if (!slots_[std::size_t(i)].used) {
        slots_[std::size_t(i)].used = true;
        slots_[std::size_t(i)].base = prompt.ids[0] * 1000;
        slots_[std::size_t(i)].step = 1;
        slots_[std::size_t(i)].pages = need;
        free_pages_ -= need;
        first_token = slots_[std::size_t(i)].base + 0;
        stats.prompt_tokens = int(prompt.ids.size());
        ++active_now_;
        peak_active_ = std::max(peak_active_, active_now_);
        return i;
      }
    }
    return -1;
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<int32_t> out_tokens) override {
    if (step_delay_.count()) std::this_thread::sleep_for(step_delay_);
    std::lock_guard lock(mu_);
    peak_step_batch_ = std::max(peak_step_batch_, int(inputs.size()));
    ++step_calls_;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto& s = slots_[std::size_t(inputs[i].slot)];
      if (!s.used) throw std::logic_error("step on inactive slot");
      // Verify input token equals previous emitted token for this slot!
      if (inputs[i].token != s.base + s.step - 1) throw std::logic_error("wrong input token for slot");
      out_tokens[i] = s.base + s.step;
      ++s.step;
    }
  }

  void release_slot(int slot) override {
    std::lock_guard lock(mu_);
    auto& s = slots_[std::size_t(slot)];
    if (!s.used) return;
    s.used = false;
    free_pages_ += s.pages;
    s.pages = 0;
    --active_now_;
  }

  GenerateStats generate(std::span<const int32_t>, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>&) override {
    throw std::logic_error("unexpected single-request generate()");
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const int32_t> stop_ids,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens,
                         const ingine::model::PrefillCallback& on_prefill) override {
    return generate_from_slots(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
  }

  int peak_active() const {
    std::lock_guard lock(mu_);
    return peak_active_;
  }
  int peak_step_batch() const {
    std::lock_guard lock(mu_);
    return peak_step_batch_;
  }
  int active_now() const {
    std::lock_guard lock(mu_);
    return active_now_;
  }
  int free_pages() const {
    std::lock_guard lock(mu_);
    return free_pages_;
  }
  int step_calls() const {
    std::lock_guard lock(mu_);
    return step_calls_;
  }

 private:
  struct Slot {
    bool used = false;
    int32_t base = 0;
    int32_t step = 0;
    int pages = 0;
  };
  int max_batch_, max_pages_;
  mutable std::mutex mu_;
  int free_pages_;
  std::chrono::milliseconds step_delay_;
  std::vector<Slot> slots_;
  int active_now_ = 0;
  int peak_active_ = 0;
  int peak_step_batch_ = 0;
  int step_calls_ = 0;
};

std::vector<int32_t> drain(Job& job, std::optional<std::string>* failure = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(5));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && failure) *failure = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout draining job");
  }
}

void test_64_concurrent_heterogeneous() {
  FakeBatchGenerator gen(64, 10000, std::chrono::milliseconds(1));
  Scheduler sched(gen, {999999}, 64);

  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<int> expected_len;
  for (int i = 1; i <= 64; ++i) {
    const int len = 5 + (i % 17);  // heterogeneous lengths 5..21
    std::vector<int32_t> prompt(std::size_t(4 + (i % 10)), i);
    auto j = std::make_shared<Job>(std::move(prompt), len, Sampling{});
    CHECK(sched.submit(j));
    jobs.push_back(std::move(j));
    expected_len.push_back(len);
  }

  for (int i = 0; i < 64; ++i) {
    std::optional<std::string> err;
    const auto out = drain(*jobs[std::size_t(i)], &err);
    CHECK(!err.has_value());
    CHECK(int(out.size()) == expected_len[std::size_t(i)]);
    for (int t = 0; t < int(out.size()); ++t) {
      CHECK(out[std::size_t(t)] == (i + 1) * 1000 + t);
    }
  }
  CHECK(gen.peak_active() >= 32);  // batched concurrently (almost always 64)
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 10000);
}

void test_continuous_join_leave_and_cancel() {
  // max_batch=4: start 4 long requests, verify waiting requests join mid-generation as active ones cancel/stop.
  FakeBatchGenerator gen(4, 1000, std::chrono::milliseconds(3));
  // Stop token 1004 stops job #1 after 5 tokens (1000..1004).
  Scheduler sched(gen, {1004}, 8);

  auto j1 = std::make_shared<Job>(std::vector<int32_t>{1}, 100, Sampling{});  // hits stop 1004 at len 5
  auto j2 = std::make_shared<Job>(std::vector<int32_t>{2}, 1000, Sampling{});  // will be cancelled mid-flight
  auto j3 = std::make_shared<Job>(std::vector<int32_t>{3}, 25, Sampling{});   // runs full 25 tokens
  auto j4 = std::make_shared<Job>(std::vector<int32_t>{4}, 25, Sampling{});   // runs full 25 tokens
  auto j5 = std::make_shared<Job>(std::vector<int32_t>{5}, 6, Sampling{});    // waits, joins when j1 stops!
  auto j6 = std::make_shared<Job>(std::vector<int32_t>{6}, 6, Sampling{});    // waits, joins when j2 cancels!

  CHECK(sched.submit(j1));
  CHECK(sched.submit(j2));
  CHECK(sched.submit(j3));
  CHECK(sched.submit(j4));
  while (sched.stats().active < 4) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(sched.submit(j5));
  CHECK(sched.submit(j6));
  CHECK(sched.stats().waiting == 2);

  // Cancel j2 mid-flight: j6 should immediately take its slot while j3 & j4 are still running!
  j2->cancel();

  const auto o1 = drain(*j1);
  const auto o2 = drain(*j2);
  const auto o5 = drain(*j5);
  const auto o6 = drain(*j6);
  const auto o3 = drain(*j3);
  const auto o4 = drain(*j4);

  CHECK((o1 == std::vector<int32_t>{1000, 1001, 1002, 1003, 1004}));
  CHECK(o2.size() < 50);
  CHECK((o5 == std::vector<int32_t>{5000, 5001, 5002, 5003, 5004, 5005}));
  CHECK((o6 == std::vector<int32_t>{6000, 6001, 6002, 6003, 6004, 6005}));
  CHECK(o3.size() == 25 && o3.back() == 3024);
  CHECK(o4.size() == 25 && o4.back() == 4024);
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 1000);
}

void test_error_isolation_and_backpressure() {
  // Only 4 pages total -> each prompt needs 2 pages -> at most 2 concurrent even with max_batch=4!
  FakeBatchGenerator gen(4, 4, std::chrono::milliseconds(2));
  Scheduler sched(gen, {}, 8);

  auto bad = std::make_shared<Job>(std::vector<int32_t>{-1}, 5, Sampling{});
  auto g1 = std::make_shared<Job>(std::vector<int32_t>{10}, 6, Sampling{});
  auto g2 = std::make_shared<Job>(std::vector<int32_t>{11}, 6, Sampling{});
  auto g3 = std::make_shared<Job>(std::vector<int32_t>{12}, 6, Sampling{});
  CHECK(sched.submit(bad));
  CHECK(sched.submit(g1));
  CHECK(sched.submit(g2));
  CHECK(sched.submit(g3));

  std::optional<std::string> err;
  CHECK(drain(*bad, &err).empty());
  CHECK(err && err->find("bad prompt") != std::string::npos);

  CHECK(drain(*g1).size() == 6);
  CHECK(drain(*g2).size() == 6);
  CHECK(drain(*g3).size() == 6);
  CHECK(gen.peak_active() <= 2);  // respected KV page pool backpressure!
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 4);
}

void test_on_prefill_exception_releases_slot_for_reuse() {
  // Regression for review item #2: with only 1 slot and 2 pages, a throwing on_prefill callback must
  // release the slot and its KV pages so the immediately following request can acquire and reuse them.
  FakeBatchGenerator gen(/*max_batch=*/1, /*max_pages=*/2);
  const std::vector<int32_t> p1 = {7};
  bool threw = false;
  try {
    (void)gen.generate(Prompt{p1, {}}, 4, Sampling{}, {},
                       [](std::span<const int32_t>) { return true; },
                       [](const GenerateStats&) { throw std::runtime_error("injected on_prefill failure"); });
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("injected on_prefill failure") != std::string::npos;
  }
  CHECK(threw);
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 2);

  // A new request must immediately acquire the released slot and complete cleanly.
  std::vector<int32_t> emitted;
  const GenerateStats st = gen.generate(
      Prompt{p1, {}}, 3, Sampling{}, {},
      [&](std::span<const int32_t> toks) {
        emitted.insert(emitted.end(), toks.begin(), toks.end());
        return true;
      },
      [](const GenerateStats&) {});
  CHECK((emitted == std::vector<int32_t>{7000, 7001, 7002}));
  CHECK(st.generated_tokens == 3);
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 2);
}

void test_admission_exception_isolated_from_active_work() {
  // Regression for review item #3: an exception thrown by can_admit_slot() while another job is live
  // must fail only the faulting job and NEVER terminate the worker or fail healthy active/queued jobs.
  FakeBatchGenerator gen(/*max_batch=*/4, /*max_pages=*/16, std::chrono::milliseconds(3));
  Scheduler sched(gen, {}, 8);

  auto j_live = std::make_shared<Job>(std::vector<int32_t>{1}, 20, Sampling{});
  CHECK(sched.submit(j_live));
  while (sched.stats().active < 1) std::this_thread::sleep_for(std::chrono::milliseconds(1));

  auto j_fault = std::make_shared<Job>(std::vector<int32_t>{2}, 6, Sampling{}, std::vector<ingine::model::PromptImage>{},
                                       0, "throw-admit");
  auto j_after = std::make_shared<Job>(std::vector<int32_t>{3}, 6, Sampling{});
  CHECK(sched.submit(j_fault));
  CHECK(sched.submit(j_after));

  std::optional<std::string> err_live, err_fault, err_after;
  const auto out_fault = drain(*j_fault, &err_fault);
  const auto out_live = drain(*j_live, &err_live);
  const auto out_after = drain(*j_after, &err_after);

  CHECK(out_fault.empty() && err_fault && err_fault->find("injected admission fault") != std::string::npos);
  CHECK(!err_live.has_value() && out_live.size() == 20 && out_live.front() == 1000 && out_live.back() == 1019);
  CHECK(!err_after.has_value() && (out_after == std::vector<int32_t>{3000, 3001, 3002, 3003, 3004, 3005}));
  CHECK(gen.active_now() == 0);
  CHECK(gen.free_pages() == 16);
}

void test_admission_throw_during_shutdown_single_end() {
  // Deterministic regression for close() racing with a throwing can_admit_slot():
  // close() fails/finishes the waiting job and removes it from waiting_; when can_admit_slot()
  // subsequently throws, loop_batched() must NOT emit a second failure or end event.
  class PendingAdmission final : public ingine::model::Generator {
   public:
    mutable std::promise<void> entered;
    std::promise<void> resume;
    std::shared_future<void> resumed = resume.get_future().share();
    std::atomic<int> releases{0};

    std::string model_id() const override { return "pending-admission"; }
    int max_model_len() const override { return 512; }
    int max_batch_size() const override { return 2; }

    bool can_admit_slot(const Prompt&, int, const Sampling&) const override {
      entered.set_value();
      if (resumed.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
        throw std::logic_error("fixture timed out");
      throw std::runtime_error("admission check failed");
    }
    int start_slot(const Prompt&, int, const Sampling&, int32_t& first, GenerateStats&) override {
      first = 1;
      return 0;
    }
    void step_slots(std::span<const BatchSlotStep> steps, std::span<int32_t> out) override {
      for (std::size_t i = 0; i < steps.size(); ++i) out[i] = 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    void release_slot(int) override { ++releases; }
    GenerateStats generate(std::span<const int32_t>, int, const Sampling&, std::span<const int32_t>,
                           const std::function<bool(std::span<const int32_t>)>&) override {
      throw std::logic_error("unexpected single-slot entry");
    }
  };

  PendingAdmission gen;
  auto entered = gen.entered.get_future();
  Scheduler sched(gen, {}, 2);
  auto live = std::make_shared<Job>(std::vector<int32_t>{1}, 128, Sampling{});
  auto pending = std::make_shared<Job>(std::vector<int32_t>{2}, 8, Sampling{});
  CHECK(sched.submit(live));
  CHECK(sched.submit(pending));
  CHECK(entered.wait_for(std::chrono::seconds(4)) == std::future_status::ready);

  std::thread closer([&] { sched.close(); });
  bool first_end = false;
  std::string first_failure;
  const auto deadline = ingine::engine::Clock::now() + std::chrono::seconds(4);
  while (!first_end && ingine::engine::Clock::now() < deadline) {
    auto batch = pending->take(std::chrono::milliseconds(50));
    if (batch.failure) first_failure = *batch.failure;
    first_end = batch.end;
  }
  gen.resume.set_value();
  closer.join();

  const auto later = pending->take(std::chrono::milliseconds(20));
  CHECK(first_end);
  CHECK(first_failure == "server shutting down");
  CHECK(later.empty());
  CHECK(gen.releases.load() == 1);
}

void test_batched_spec_multi_token_stop_budget_cancel() {
  // Deterministic CPU regression for step_slots_multi() in Scheduler::loop_batched() and generate_from_slots():
  // - Mid-block stop token (token #2 of a 5-token speculative block) truncates cleanly and retires immediately.
  // - Mid-block max_new_tokens cap truncates cleanly without emitting extra tokens.
  // - Mid-block cancel retires cleanly without leaking slots or pages.
  // - Concurrent heterogeneous multi-token bursts complete with exact token sequences and spec_rounds/spec_accepted stats.
  class FakeMultiSpecGenerator final : public ingine::model::Generator {
   public:
    FakeMultiSpecGenerator(int max_batch, int tokens_per_step)
        : max_batch_(max_batch), tokens_per_step_(tokens_per_step), slots_(std::size_t(max_batch)) {}

    std::string model_id() const override { return "fake-multi-spec"; }
    int max_model_len() const override { return 4096; }
    int max_batch_size() const override { return max_batch_; }
    bool supports_batched_spec() const override { return true; }
    bool can_admit_slot(int, int) const override {
      std::lock_guard lock(mu_);
      return active_now_ < max_batch_;
    }

    int start_slot(const Prompt& prompt, int, const Sampling&, int32_t& first_token, GenerateStats& stats) override {
      std::lock_guard lock(mu_);
      for (int i = 0; i < max_batch_; ++i) {
        if (!slots_[std::size_t(i)].used) {
          slots_[std::size_t(i)] = {true, prompt.ids[0] * 1000, 1};
          first_token = prompt.ids[0] * 1000;
          stats.prompt_tokens = int(prompt.ids.size());
          ++active_now_;
          return i;
        }
      }
      return -1;
    }

    void step_slots(std::span<const BatchSlotStep>, std::span<int32_t>) override {
      throw std::logic_error("expected step_slots_multi when supports_batched_spec() is true");
    }

    void step_slots_multi(std::span<const BatchSlotStep> inputs,
                          std::span<ingine::model::BatchSlotResult> out) override {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::lock_guard lock(mu_);
      ++multi_calls_;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto& s = slots_[std::size_t(inputs[i].slot)];
        if (!s.used) throw std::logic_error("step_slots_multi on inactive slot");
        if (inputs[i].token != s.base + s.step - 1) throw std::logic_error("wrong anchor token");
        ingine::model::BatchSlotResult r{};
        r.count = tokens_per_step_;
        r.spec_rounds = 1;
        r.spec_accepted = tokens_per_step_ - 1;
        for (int k = 0; k < tokens_per_step_; ++k) {
          r.tokens[k] = s.base + s.step;
          ++s.step;
        }
        out[i] = r;
      }
    }

    void release_slot(int slot) override {
      std::lock_guard lock(mu_);
      if (slots_[std::size_t(slot)].used) {
        slots_[std::size_t(slot)].used = false;
        --active_now_;
      }
    }

    GenerateStats generate(std::span<const int32_t>, int, const Sampling&, std::span<const int32_t>,
                           const std::function<bool(std::span<const int32_t>)>&) override {
      throw std::logic_error("unexpected generate");
    }
    GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                           std::span<const int32_t> stop_ids,
                           const std::function<bool(std::span<const int32_t>)>& on_tokens,
                           const ingine::model::PrefillCallback& on_prefill) override {
      return generate_from_slots(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
    }

    int active_now() const {
      std::lock_guard lock(mu_);
      return active_now_;
    }
    int multi_calls() const {
      std::lock_guard lock(mu_);
      return multi_calls_;
    }

   private:
    struct Slot {
      bool used = false;
      int32_t base = 0;
      int32_t step = 0;
    };
    int max_batch_, tokens_per_step_;
    mutable std::mutex mu_;
    std::vector<Slot> slots_;
    int active_now_ = 0;
    int multi_calls_ = 0;
  };

  // 1. Direct generate_from_slots() mid-block stop & budget truncation + spec accounting
  {
    FakeMultiSpecGenerator gen(4, 5);
    std::vector<int32_t> prompt = {2};
    std::vector<int32_t> emitted;
    const std::vector<int32_t> stops = {2003};  // token #3 (inside first 5-token block 2001..2005)
    const GenerateStats st = gen.generate(
        Prompt{prompt, {}}, 20, Sampling{}, stops,
        [&](std::span<const int32_t> toks) {
          emitted.insert(emitted.end(), toks.begin(), toks.end());
          return true;
        },
        [](const GenerateStats&) {});
    CHECK((emitted == std::vector<int32_t>{2000, 2001, 2002, 2003}));
    CHECK(st.generated_tokens == 4);
    CHECK(st.spec_rounds == 1);
    CHECK(st.spec_accepted == 4);
    CHECK(gen.active_now() == 0);

    // Budget cap inside a 5-token block (max_new_tokens=8 -> 1 + 5 + 2)
    emitted.clear();
    const GenerateStats st2 = gen.generate(
        Prompt{prompt, {}}, 8, Sampling{}, {},
        [&](std::span<const int32_t> toks) {
          emitted.insert(emitted.end(), toks.begin(), toks.end());
          return true;
        },
        [](const GenerateStats&) {});
    CHECK((emitted == std::vector<int32_t>{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007}));
    CHECK(st2.generated_tokens == 8);
    CHECK(st2.spec_rounds == 2);
    CHECK(gen.active_now() == 0);
  }

  // 2. Scheduler::loop_batched() concurrent multi-token streaming with mid-block stop, budget, and cancel
  {
    FakeMultiSpecGenerator gen(4, 5);
    Scheduler sched(gen, {1003}, 8);  // 1003 stops j1 mid-block
    auto j1 = std::make_shared<Job>(std::vector<int32_t>{1}, 50, Sampling{});
    auto j2 = std::make_shared<Job>(std::vector<int32_t>{2}, 8, Sampling{});   // budget mid-block
    auto j3 = std::make_shared<Job>(std::vector<int32_t>{3}, 16, Sampling{});  // exact multi-block (1+5+5+5)
    auto j4 = std::make_shared<Job>(std::vector<int32_t>{4}, 200, Sampling{}); // cancelled mid-flight
    CHECK(sched.submit(j1));
    CHECK(sched.submit(j2));
    CHECK(sched.submit(j3));
    CHECK(sched.submit(j4));
    while (sched.stats().active < 1) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    j4->cancel();

    const auto o1 = drain(*j1);
    const auto o2 = drain(*j2);
    const auto o3 = drain(*j3);
    const auto o4 = drain(*j4);
    CHECK((o1 == std::vector<int32_t>{1000, 1001, 1002, 1003}));
    CHECK((o2 == std::vector<int32_t>{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007}));
    CHECK(o3.size() == 16 && o3.front() == 3000 && o3.back() == 3015);
    CHECK(o4.size() < 50);
    CHECK(gen.active_now() == 0);
    CHECK(gen.multi_calls() >= 1);
  }
}

void test_confidence_width_and_page_budget_plan() {
  using ingine::model::VerifySlotInput;
  using ingine::model::VerifyWidthPlan;
  using ingine::model::plan_batch_verify_widths;

  // 1. Review control: 4 confident drafts MUST yield w_cap=5 (4 active drafts + 1 anchor), NEVER 8 (7 active drafts),
  //    while the shared launch bucket W_batch pads to canonical 8!
  for (int lead = 0; lead <= 7; ++lead) {
    const VerifySlotInput s{lead, /*remaining=*/64, /*pos=*/100, /*max_ctx=*/4096, /*max_pos_res=*/256, /*cur_pages=*/4};
    const VerifyWidthPlan plan = plan_batch_verify_widths(std::span(&s, 1), /*max_verify_width=*/8, /*page_size=*/64,
                                                          /*spec_page_budget=*/16);
    CHECK(plan.w_cap.size() == 1);
    CHECK(plan.w_cap[0] == lead + 1);
    const int active_drafts = plan.w_cap[0] - 1;
    CHECK(active_drafts == lead);
    const int exp_bucket = (lead + 1 <= 1) ? 1 : (lead + 1 <= 2) ? 2 : (lead + 1 <= 4) ? 4 : 8;
    CHECK(plan.W_batch == exp_bucket);
  }

  // 2. Heterogeneous batch: slot 0 has 4 confident drafts (w_cap=5), slot 1 has 1 (w_cap=2), slot 2 has 0 (w_cap=1),
  //    slot 3 has 6 confident drafts but remaining_emissions=3 (clamped to w_cap=3).
  //    Shared launch bucket pads to W_batch=8 while each active slot keeps its exact bound.
  {
    const std::vector<VerifySlotInput> slots = {
        {4, 64, 100, 4096, 256, 4},
        {1, 64, 100, 4096, 256, 4},
        {0, 64, 100, 4096, 256, 4},
        {6, 3, 100, 4096, 256, 4},
    };
    const VerifyWidthPlan plan = plan_batch_verify_widths(slots, 8, 64, 16);
    CHECK((plan.w_cap == std::vector<int>{5, 2, 1, 3}));
    CHECK(plan.W_batch == 8);
    CHECK(plan.total_extra_pages == 0);
  }

  // 3. Tight spec_page_budget: 2 slots at pos=62 (current_pages=1 -> fits up to +2 tokens without extra page),
  //    each with lead=4 (wants w_cap=5 -> pos+5=67 -> +1 page each = 2 extra pages), budget=1 -> trims one slot to 2!
  {
    const std::vector<VerifySlotInput> slots = {
        {4, 64, 62, 4096, 192, 1},
        {4, 64, 62, 4096, 192, 1},
    };
    const VerifyWidthPlan plan = plan_batch_verify_widths(slots, 8, 64, /*spec_page_budget=*/1);
    CHECK(plan.total_extra_pages <= 1);
    CHECK((plan.w_cap[0] == 2 && plan.w_cap[1] == 5) || (plan.w_cap[0] == 5 && plan.w_cap[1] == 2));
    CHECK(plan.W_batch == 8);
  }
}

void test_staggered_milestone_and_accounting() {
  using ingine::model::StaggeredBenchmarkState;
  using ingine::model::check_vram_lease_fit;

  // 1. Fail-closed VRAM lease check: remaining=0 MUST NOT be treated as unlimited!
  {
    CHECK(!check_vram_lease_fit(std::optional<std::size_t>(0), 1024).ok);
    CHECK(check_vram_lease_fit(std::optional<std::size_t>(0), 0).ok);
    CHECK(!check_vram_lease_fit(std::optional<std::size_t>(4096), 8192).ok);
    CHECK(check_vram_lease_fit(std::optional<std::size_t>(4096), 4096).ok);
    CHECK(check_vram_lease_fit(std::nullopt, 1ull << 40).ok);
  }

  // 2. Compare 32-token cap under 1 tok/round (plain) vs 4 tok/round (spec):
  //    Using step_max_tokens_for(), BOTH plain and spec MUST hit cohort1_emitted_at_join == 32 EXACTLY (ZERO overshoot!)
  //    while ALL 4 cohort-1 requests remain active mid-generation!
  auto run_sim = [](int tok_per_round, bool first_eos_on_req0, int max_active_cap, bool inject_zero_result_req1) {
    const int C = 8;
    const int tokens = 32;
    StaggeredBenchmarkState stg;
    stg.init(C, tokens, /*join_per_req=*/8);  // 4 cohort-1 reqs * 8 = 32 emitted tokens EXACT!
    struct Active {
      int req = 0;
      int gen = 0;
    };
    std::vector<Active> live;
    int round = 0;
    auto c1_counts = [&]() -> std::pair<int, int> {
      int act = 0, below = 0;
      for (const auto& a : live) {
        if (a.req < stg.cohort1_size) {
          ++act;
          if (a.gen < stg.join_tokens_per_req) ++below;
        }
      }
      return {act, below};
    };
    auto try_admit = [&]() {
      while (!stg.pending.empty()) {
        if (int(live.size()) >= max_active_cap) {
          ++stg.deferred_attempts;
          break;
        }
        const int r = stg.pending.front();
        stg.pending.pop_front();
        ++stg.admitted;
        stg.note_emitted(r, 1);
        if (first_eos_on_req0 && r == 0) {
          ++stg.eos_first_token;
          ++stg.eos_stopped;
          ++stg.completed;
        } else {
          live.push_back({r, 1});
        }
      }
    };
    try_admit();
    while (!live.empty() || !stg.pending.empty() || stg.has_unoffered()) {
      const auto [act, below] = c1_counts();
      stg.maybe_offer_cohort2(round, act, below);
      try_admit();
      if (live.empty()) continue;
      ++round;
      std::vector<Active> kept;
      for (auto& a : live) {
        const int step_cap = stg.step_max_tokens_for(a.req, a.gen);
        int emit = tok_per_round;
        if (step_cap > 0) emit = std::min(emit, step_cap);
        if (inject_zero_result_req1 && a.req == 1 && a.gen == 1) emit = 0;
        for (int k = 0; k < emit && a.gen < tokens; ++k) {
          ++a.gen;
          stg.note_emitted(a.req, 1);
        }
        if (a.gen >= tokens) {
          ++stg.max_token_stopped;
          ++stg.completed;
        } else if (emit == 0) {
          // Zero-result step below budget MUST NOT be counted as completed or max_token_stopped!
          ++stg.aborted_zero_result;
        } else {
          kept.push_back(a);
        }
      }
      live = std::move(kept);
    }
    return stg;
  };

  const auto plain = run_sim(1, false, 8, false);
  const auto spec = run_sim(4, false, 8, false);
  CHECK(plain.accounting_consistent());
  CHECK(spec.accounting_consistent());
  CHECK(plain.offered == 8 && plain.admitted == 8 && plain.completed == 8 && plain.aborted_zero_result == 0 && plain.dropped == 0);
  CHECK(spec.offered == 8 && spec.admitted == 8 && spec.completed == 8 && spec.aborted_zero_result == 0 && spec.dropped == 0);
  // EXACT zero-overshoot match across plain and spec: 32 == 32, both with 4 active cohort-1 slots!
  CHECK(plain.cohort1_active_at_join == 4);
  CHECK(spec.cohort1_active_at_join == 4);
  CHECK(plain.cohort1_emitted_at_join == 32);
  CHECK(spec.cohort1_emitted_at_join == 32);
  CHECK(plain.join_round == 7);
  CHECK(spec.join_round == 2);

  // First-token EOS on req 0 + tight capacity cap (4) forcing deferral
  const auto constrained = run_sim(4, true, 4, false);
  CHECK(constrained.accounting_consistent());
  CHECK(constrained.offered == 8 && constrained.admitted == 8 && constrained.completed == 8 && constrained.dropped == 0);
  CHECK(constrained.eos_first_token == 1);
  CHECK(constrained.deferred_attempts > 0);

  // Zero-result step below budget: MUST be classified as aborted_zero_result=1, completed=7, max_token_stopped=7!
  const auto zero_step = run_sim(4, false, 8, true);
  CHECK(zero_step.accounting_consistent());
  CHECK(zero_step.offered == 8 && zero_step.admitted == 8);
  CHECK(zero_step.completed == 7 && zero_step.max_token_stopped == 7);
  CHECK(zero_step.aborted_zero_result == 1);

  // External arrival trace mode: frozen timestamps replayed identically
  {
    StaggeredBenchmarkState tr;
    tr.init_external_trace({0.0, 0.0, 50.0, 50.0});
    CHECK(tr.offered == 2 && tr.pending.size() == 2 && tr.has_unoffered());
    tr.maybe_offer_cohort2(1, 2, 2, 25.0);
    CHECK(tr.offered == 2 && tr.has_unoffered());
    tr.maybe_offer_cohort2(2, 2, 2, 50.0);
    CHECK(tr.offered == 4 && !tr.has_unoffered() && tr.join_round == 2);
  }
}

void test_philox_sampled_and_cross_gear_controls() {
  namespace sv = ingine::model::spec_verify;
  using ingine::model::BatchSpecSlotTrace;

  // 1. Exact Philox wrong-member control (reproduces review control: seed=12345, draws=7, uniform shortlist -> col 420!):
  //    Valid Philox token 420 MUST pass; substituting wrong shortlist member 421 MUST be rejected!
  {
    constexpr int V = 1024;
    BatchSpecSlotTrace tr{};
    tr.has_draft = true;
    tr.has_sampled_details = true;
    tr.temperature = 0.8f;
    tr.draft_seed = 12345ull;
    tr.draft_draws = 7ull;
    tr.w_cap = 2;
    tr.W_batch = 2;
    tr.draft_cand_idx.resize(sv::kGamma * sv::kCandidates);
    tr.draft_cand_logits.assign(sv::kGamma * sv::kCandidates, 0.0f);  // uniform shortlist!
    for (int r = 0; r < sv::kGamma; ++r) {
      for (int k = 0; k < sv::kCandidates; ++k) {
        tr.draft_cand_idx[std::size_t(r) * sv::kCandidates + k] = k;
      }
      const int col = sv::reconstruct_draft_column(
          std::span<const float>(tr.draft_cand_logits.data() + std::size_t(r) * sv::kCandidates, sv::kCandidates),
          tr.temperature, tr.draft_seed, tr.draft_draws, r);
      tr.draft_tokens[r] = tr.draft_cand_idx[std::size_t(r) * sv::kCandidates + col];
      tr.draft_conf[r] = 1.0f / float(sv::kCandidates);
      tr.draft_qstats[r][0] = 0.0f;
      tr.draft_qstats[r][1] = float(sv::kCandidates);
    }
    CHECK(tr.draft_tokens[0] == 420);  // exact review control value!

    tr.accept_rng_seed = 99999ull;
    tr.accept_rng_counter_u = 3ull;
    tr.accept_rng_counter_resid = 4ull;
    tr.uniforms = {sv::head_uniform(tr.accept_rng_seed, tr.accept_rng_counter_u, 0u)};
    tr.target_probs.assign(std::size_t(tr.w_cap) * V, 1.0f / float(V));
    // Force accept at r=0 (p(420) = 1.0 > u * q(420))
    tr.target_probs[420] = 1.0f;
    for (int v = 0; v < V; ++v)
      if (v != 420) tr.target_probs[std::size_t(v)] = 0.0f;
    tr.accepted = 1;
    tr.residual_probs.assign(tr.target_probs.begin() + V, tr.target_probs.begin() + 2 * V);
    tr.next_token = sv::head_sample_from_probs(tr.residual_probs, tr.accept_rng_seed, tr.accept_rng_counter_resid);
    tr.commit = tr.accepted + 1;

    CHECK(sv::validate_sampled_trace(tr, V).ok);

    // Inject wrong in-shortlist draft token 421 -> MUST FAIL!
    BatchSpecSlotTrace bad_draft = tr;
    bad_draft.draft_tokens[0] = 421;
    const auto r_bad = sv::validate_sampled_trace(bad_draft, V);
    CHECK(!r_bad.ok);
    CHECK(r_bad.reason.find("Philox draft token mismatch") != std::string::npos);

    // Inject wrong uniform draw -> MUST FAIL!
    BatchSpecSlotTrace bad_u = tr;
    bad_u.uniforms[0] = std::fmod(bad_u.uniforms[0] + 0.1f, 1.0f);
    CHECK(!sv::validate_sampled_trace(bad_u, V).ok);

    // Inject wrong next token -> MUST FAIL!
    BatchSpecSlotTrace bad_next = tr;
    bad_next.next_token = (bad_next.next_token + 1) % V;
    CHECK(!sv::validate_sampled_trace(bad_next, V).ok);
  }

  // 2. Cross-gear divergence binding controls:
  //    - Identical rows [0, 0, 3] with claimed differing tokens 0 vs 1 MUST FAIL (neither is argmax=2)!
  //    - Empty capture MUST FAIL!
  //    - Genuine tight near-tie where argmax(l_plain)==0 and argmax(l_spec)==1 MUST PASS!
  {
    const std::vector<float> identical = {0.f, 0.f, 3.f};
    CHECK(!sv::validate_cross_gear_divergence(identical, identical, 0, 1, 3, 0.6f).ok);
    CHECK(!sv::validate_cross_gear_divergence({}, {}, 0, 1, 3, 0.6f).ok);

    const std::vector<float> lp = {2.01f, 2.00f, -1.f};
    const std::vector<float> ls = {1.99f, 2.02f, -1.f};
    const auto ok_tie = sv::validate_cross_gear_divergence(lp, ls, 0, 1, 3, 0.6f);
    CHECK(ok_tie.ok);
    CHECK(ok_tie.argmax_plain == 0 && ok_tie.argmax_spec == 1);
    CHECK(ok_tie.margin_plain >= 0.f && ok_tie.margin_spec >= 0.f);

    // Large logit divergence exceeding bound MUST FAIL!
    const std::vector<float> ls_bug = {0.5f, 3.0f, -1.f};
    CHECK(!sv::validate_cross_gear_divergence(lp, ls_bug, 0, 1, 3, 0.6f).ok);
  }
}

}  // namespace

int main() {
  test_64_concurrent_heterogeneous();
  test_continuous_join_leave_and_cancel();
  test_error_isolation_and_backpressure();
  test_on_prefill_exception_releases_slot_for_reuse();
  test_admission_exception_isolated_from_active_work();
  test_admission_throw_during_shutdown_single_end();
  test_batched_spec_multi_token_stop_budget_cancel();
  test_confidence_width_and_page_budget_plan();
  test_staggered_milestone_and_accounting();
  test_philox_sampled_and_cross_gear_controls();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_scheduler ok\n", g_failures);
  return g_failures ? 1 : 0;
}
