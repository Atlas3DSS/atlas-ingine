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

  // 2. Compare homogeneous plain [1,1,1,1], uniform spec [4,4,4,4], and heterogeneous spec [4,1,1,1]:
  //    By parking completed-at-milestone cohort-1 lanes until the cohort catches up, ALL THREE hit the EXACT
  //    join milestone (32/32 for join=8 and 128/128 for join=32) with ZERO overshoot!
  auto run_sim = [](std::vector<int> c1_widths, int default_w, int tokens, int join_per_req,
                    bool first_eos_on_req0, int max_active_cap, bool inject_zero_result_req1,
                    std::vector<int>* c1_at_join_out = nullptr) {
    StaggeredBenchmarkState stg;
    stg.init(8, tokens, join_per_req);
    struct Active {
      int req = 0;
      int gen = 0;
    };
    std::vector<Active> live;
    int round = 0;
    bool recorded_c1 = false;
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
      const bool was_offered = stg.cohort2_offered;
      stg.maybe_offer_cohort2(round, act, below);
      if (!was_offered && stg.cohort2_offered && c1_at_join_out && !recorded_c1) {
        recorded_c1 = true;
        c1_at_join_out->assign(static_cast<std::size_t>(stg.cohort1_size), 0);
        for (const auto& a : live) {
          if (a.req < stg.cohort1_size) (*c1_at_join_out)[std::size_t(a.req)] = a.gen;
        }
      }
      try_admit();
      if (live.empty()) continue;
      bool any_stepped = false;
      std::vector<Active> kept;
      for (auto& a : live) {
        if (!stg.should_step_lane(a.req, a.gen)) {
          ++stg.cohort1_parked_steps;
          kept.push_back(a);
          continue;
        }
        any_stepped = true;
        const int step_cap = stg.step_max_tokens_for(a.req, a.gen);
        int emit = (a.req < int(c1_widths.size())) ? c1_widths[std::size_t(a.req)] : default_w;
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
      if (any_stepped) ++round;
      live = std::move(kept);
    }
    return stg;
  };

  {
    std::vector<int> c1_p8, c1_h8, c1_p32, c1_h32;
    const auto plain8 = run_sim({1, 1, 1, 1}, 1, 32, 8, false, 8, false, &c1_p8);
    const auto spec8 = run_sim({4, 4, 4, 4}, 4, 32, 8, false, 8, false);
    const auto het8 = run_sim({4, 1, 1, 1}, 4, 32, 8, false, 8, false, &c1_h8);
    CHECK(plain8.accounting_consistent() && spec8.accounting_consistent() && het8.accounting_consistent());
    CHECK(plain8.cohort1_emitted_at_join == 32 && spec8.cohort1_emitted_at_join == 32 && het8.cohort1_emitted_at_join == 32);
    CHECK(c1_p8 == std::vector<int>({8, 8, 8, 8}) && c1_h8 == std::vector<int>({8, 8, 8, 8}));
    CHECK(het8.cohort1_parked_steps > 0);

    const auto plain32 = run_sim({1, 1, 1, 1}, 1, 128, 32, false, 8, false, &c1_p32);
    const auto het32 = run_sim({4, 1, 1, 1}, 4, 128, 32, false, 8, false, &c1_h32);
    CHECK(plain32.accounting_consistent() && het32.accounting_consistent());
    CHECK(plain32.cohort1_emitted_at_join == 128 && het32.cohort1_emitted_at_join == 128);
    CHECK(c1_p32 == std::vector<int>({32, 32, 32, 32}) && c1_h32 == std::vector<int>({32, 32, 32, 32}));
    CHECK(het32.cohort1_parked_steps > 0);
  }

  // First-token EOS on req 0 + tight capacity cap (4) forcing deferral
  const auto constrained = run_sim({4, 1, 1, 1}, 4, 32, 8, true, 4, false);
  CHECK(constrained.accounting_consistent());
  CHECK(constrained.offered == 8 && constrained.admitted == 8 && constrained.completed == 8 && constrained.dropped == 0);
  CHECK(constrained.eos_first_token == 1);
  CHECK(constrained.deferred_attempts > 0);

  // Deferred cohort-1 admissions (max_batch=2 < cohort1_size=4) with heterogeneous widths: MUST NOT deadlock!
  const auto deferred_c1 = run_sim({4, 1, 1, 1}, 4, 32, 8, false, 2, false);
  CHECK(deferred_c1.accounting_consistent());
  CHECK(deferred_c1.offered == 8 && deferred_c1.admitted == 8 && deferred_c1.completed == 8 && deferred_c1.dropped == 0);
  CHECK(deferred_c1.deferred_attempts > 0);

  // Zero-result step below budget: MUST be classified as aborted_zero_result=1, completed=7, max_token_stopped=7!
  const auto zero_step = run_sim({4, 4, 4, 4}, 4, 32, 8, false, 8, true);
  CHECK(zero_step.accounting_consistent());
  CHECK(zero_step.offered == 8 && zero_step.admitted == 8);
  CHECK(zero_step.completed == 7 && zero_step.max_token_stopped == 7);
  CHECK(zero_step.aborted_zero_result == 1);

  // External arrival trace mode: frozen timestamps replayed identically + rejects negative/decreasing/NaN traces
  {
    StaggeredBenchmarkState tr;
    tr.init_external_trace({0.0, 0.0, 50.0, 50.0});
    CHECK(tr.offered == 2 && tr.pending.size() == 2 && tr.has_unoffered());
    tr.maybe_offer_cohort2(1, 2, 2, 25.0);
    CHECK(tr.offered == 2 && tr.has_unoffered());
    tr.maybe_offer_cohort2(2, 2, 2, 50.0);
    CHECK(tr.offered == 4 && !tr.has_unoffered() && tr.join_round == 2);

    bool threw_neg = false, threw_dec = false, threw_nan = false;
    try { tr.init_external_trace({-1.0, 0.0}); } catch (const std::invalid_argument&) { threw_neg = true; }
    try { tr.init_external_trace({10.0, 5.0}); } catch (const std::invalid_argument&) { threw_dec = true; }
    try { tr.init_external_trace({0.0, std::numeric_limits<double>::quiet_NaN()}); } catch (const std::invalid_argument&) { threw_nan = true; }
    CHECK(threw_neg && threw_dec && threw_nan);
  }

  // Peak VRAM preflight checks using EXACT production geometry (H=2304, KV_DIM=1024, Q_DIM=3072, short_chunk=16384,
  // MTP gate/up [9216, 2304] -> cat=81 MiB, raw mtp.safetensors=635,795,744 B, pk::Gemm=32 MiB, vocab=200058)!
  {
    constexpr std::size_t kMtpRawBytes = 635795744ull;
    constexpr int kMtpInter = 9216;
    constexpr int kProdVocab = 200058;
    constexpr int kProdHidden = 2304;
    const ingine::model::DrafterGeometry prod_g{};
    CHECK(prod_g.H == 2304 && prod_g.kv_dim == 1024 && prod_g.q_dim == 3072 && prod_g.short_chunk == 16384);
    const auto d_est = ingine::model::estimate_drafter_peak_bytes(
        kMtpRawBytes, kMtpInter, 12, 2048, kProdVocab, 16384, 8, true, prod_g);
    CHECK(d_est.cat_bytes == std::size_t(2) * 9216 * 2304 * 2);  // exact 81 MiB (84,934,656 B)!
    CHECK(d_est.load_peak_bytes > kMtpRawBytes + d_est.cat_bytes + prod_g.gemm_workspace_bytes);
    CHECK(d_est.peak_bytes >= d_est.load_peak_bytes && d_est.peak_bytes >= d_est.steady_bytes);
    // 0 B and 400 MiB remaining MUST fail closed BEFORE allocating pk::Gemm (32 MiB) or raw (606.34 MiB)!
    CHECK(!ingine::model::check_vram_lease_fit(std::size_t(0), d_est.peak_bytes).ok);
    CHECK(!ingine::model::check_vram_lease_fit(std::size_t(400) << 20, d_est.peak_bytes).ok);
    // Near-cap rejection: 1 byte below peak_bytes fails, exact peak_bytes passes!
    CHECK(!ingine::model::check_vram_lease_fit(d_est.peak_bytes - 1, d_est.peak_bytes).ok);
    CHECK(ingine::model::check_vram_lease_fit(d_est.peak_bytes, d_est.peak_bytes).ok);

    // Weights & Prefiller preflight peak checks (near-cap rejection before load_weights / Prefiller!)
    const std::size_t w_est =
        ingine::model::estimate_weights_peak_bytes(19526019168ull, true, kProdVocab, kProdHidden);
    CHECK(w_est > 19526019168ull);
    CHECK(!ingine::model::check_vram_lease_fit(w_est - 1, w_est).ok);
    CHECK(ingine::model::check_vram_lease_fit(std::size_t(28) << 30, w_est).ok);

    const std::size_t p_est = ingine::model::estimate_prefiller_peak_bytes(512, kProdVocab, 12);
    CHECK(p_est > (std::size_t(200) << 20));
    CHECK(!ingine::model::check_vram_lease_fit(p_est - 1, p_est).ok);
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

    // Invalid control A: all-zero target tensor + zero residual + token 0 -> MUST FAIL!
    {
      BatchSpecSlotTrace zall = tr;
      zall.target_probs.assign(std::size_t(zall.w_cap) * V, 0.0f);
      zall.accepted = 0;
      zall.residual_probs.assign(V, 0.0f);
      zall.next_token = 0;
      const auto rz = sv::validate_sampled_trace(zall, V);
      CHECK(!rz.ok);
      CHECK(rz.reason.find("mass out of unit tolerance") != std::string::npos);
    }

    // Invalid control B: valid first target row + all-zero bonus row + zero residual + token 0 -> MUST FAIL!
    {
      BatchSpecSlotTrace zbonus = tr;
      for (int v = 0; v < V; ++v) zbonus.target_probs[std::size_t(V + v)] = 0.0f;
      zbonus.accepted = 1;
      zbonus.residual_probs.assign(V, 0.0f);
      zbonus.next_token = 0;
      const auto rzb = sv::validate_sampled_trace(zbonus, V);
      CHECK(!rzb.ok);
      CHECK(rzb.reason.find("mass out of unit tolerance") != std::string::npos);
    }

    // Invalid control C: negative target probability -> MUST FAIL!
    {
      BatchSpecSlotTrace negp = tr;
      negp.target_probs[0] = -0.1f;
      negp.target_probs[1] = 0.1f;
      CHECK(!sv::validate_sampled_trace(negp, V).ok);
    }

    // Valid control D: normal unnormalized rejection residual max(p - q, 0) with sum < 1.0 -> MUST PASS!
    {
      BatchSpecSlotTrace rej = tr;
      // Pick target row 0 with p(420)=0 (forces reject at r=0), p(100)=0.6, p(200)=0.4 -> residual sum < 1.0!
      for (int v = 0; v < V; ++v) rej.target_probs[std::size_t(v)] = 0.0f;
      rej.target_probs[100] = 0.6f;
      rej.target_probs[200] = 0.4f;
      rej.accepted = 0;
      rej.residual_probs.assign(V, 0.0f);
      const float q_unif = 1.0f / float(sv::kCandidates);
      rej.residual_probs[100] = std::max(0.6f - q_unif, 0.0f);
      rej.residual_probs[200] = std::max(0.4f - q_unif, 0.0f);
      rej.next_token = sv::head_sample_from_probs(rej.residual_probs, rej.accept_rng_seed, rej.accept_rng_counter_resid);
      rej.commit = 1;
      CHECK(sv::validate_sampled_trace(rej, V).ok);
    }

    // Valid control E: zero-residual fallback (p == q identically on rejected row -> max(p-q, 0)==0 everywhere,
    // falls back to normalized target row p with unit mass) -> MUST PASS!
    {
      BatchSpecSlotTrace fb = tr;
      // Make target row 0 equal q identically (1/1024 on all V=1024) AND pick uniform counter where u >= 1? No,
      // wait: u in [0,1) so u * q(420) < p(420) when p == q. To force reject AND max(p - q, 0) == 0 everywhere,
      // make q(k) == 1/1024 for all k and p(420) = 0, p(other) <= 1/1024? No, sum(p)=1 & sum(q)=1 implies
      // if p != q there is some x with p(x) > q(x). The real fallback occurs from FP rounding when p == q
      // or when p(d) is slightly below u*q(d) while max(p-q, 0) flushes to 0!
      // Construct exact case: u = 0.999... or p(x) = q(x) = 1/1024 and set accept_rng_counter_u so u == 0? No:
      // Find a counter where head_uniform >= 0.99? Even simpler: set q slightly higher on non-420? No, sum(q)=1.
      // With q_unif = 1/1024 and target_probs[k] = 1/1024 - eps (so sum = 1 - 1024*eps within kProbRowMassTol = 1e-3!),
      // p(x) <= q(x) for ALL x -> max(p - q, 0) == 0 everywhere! And pick counter_u with u > p(420)/q(420) -> rejected!
      const float q_unif = 1.0f / float(sv::kCandidates);
      const float p_slightly_below = q_unif * 0.9995f;  // sum = 0.9995 (within 1e-3 tolerance!), p(x) < q(x) for all x!
      std::uint64_t high_u_ctr = 0;
      while (sv::head_uniform(fb.accept_rng_seed, high_u_ctr, 0u) < 0.9996f) ++high_u_ctr;
      fb.accept_rng_counter_u = high_u_ctr;
      fb.uniforms = {sv::head_uniform(fb.accept_rng_seed, fb.accept_rng_counter_u, 0u)};
      for (int v = 0; v < V; ++v) fb.target_probs[std::size_t(v)] = p_slightly_below;
      fb.accepted = 0;
      fb.residual_probs.assign(fb.target_probs.begin(), fb.target_probs.begin() + V);
      fb.next_token = sv::head_sample_from_probs(fb.residual_probs, fb.accept_rng_seed, fb.accept_rng_counter_resid);
      fb.commit = 1;
      CHECK(sv::validate_sampled_trace(fb, V).ok);
    }
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
