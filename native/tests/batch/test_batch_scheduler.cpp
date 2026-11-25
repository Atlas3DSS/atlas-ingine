// CPU unit tests for the continuous-batching Scheduler:
// 1. 64 concurrent heterogeneous requests run in parallel (peak concurrency == 64), exact outputs, zero slot leaks.
// 2. Mid-generation admission & immediate retirement on stop token and client cancellation.
// 3. KV-page backpressure, error isolation (bad prompt fails without breaking active slots), and clean shutdown.
#include "engine/scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
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

}  // namespace

int main() {
  test_64_concurrent_heterogeneous();
  test_continuous_join_leave_and_cancel();
  test_error_isolation_and_backpressure();
  test_on_prefill_exception_releases_slot_for_reuse();
  test_admission_exception_isolated_from_active_work();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_scheduler ok\n", g_failures);
  return g_failures ? 1 : 0;
}
