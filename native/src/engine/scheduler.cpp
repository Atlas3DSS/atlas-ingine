#include "engine/scheduler.h"

#include "engine/log.h"

#include <cxxabi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <span>
#include <typeinfo>
#include <utility>

namespace ingine::engine {
namespace {

// uuid.uuid4().hex: 32 random hex digits.
std::string random_hex32() {
  thread_local std::mt19937_64 rng{std::random_device{}() ^ (uint64_t(std::random_device{}()) << 32)};
  char buf[33];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return buf;
}

std::string type_name(const std::exception& e) {
  int status = 0;
  char* name = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
  std::string out = status == 0 && name ? name : typeid(e).name();
  std::free(name);
  return out;
}

}  // namespace

Job::Job(std::vector<int32_t> prompt_ids_, int max_new_tokens_, model::Sampling sampling_,
         std::vector<model::PromptImage> images_, int root_prefix_tokens_, std::string cache_namespace_,
         bool bypass_prefix_cache_)
    : id("chatcmpl-" + random_hex32()),
      prompt_ids(std::move(prompt_ids_)),
      images(std::move(images_)),
      root_prefix_tokens(root_prefix_tokens_),
      cache_namespace(std::move(cache_namespace_)),
      bypass_prefix_cache(bypass_prefix_cache_),
      max_new_tokens(max_new_tokens_),
      sampling(sampling_),
      t_submit(Clock::now()) {}

std::optional<Clock::time_point> Job::t_start() const {
  std::lock_guard lock(mutex_);
  return t_start_;
}

std::optional<model::GenerateStats> Job::prefill_stats() const {
  std::lock_guard lock(mutex_);
  return prefill_stats_;
}

std::optional<model::GenerateStats> Job::wait_prefill_stats() const {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return prefill_stats_.has_value() || finished_ || failure_.has_value(); });
  return prefill_stats_;
}

JobBatch Job::take(Clock::duration timeout) {
  std::unique_lock lock(mutex_);
  cv_.wait_for(lock, timeout, [&] { return !tokens_.empty() || failure_ || end_; });
  JobBatch batch;
  batch.tokens.assign(tokens_.begin(), tokens_.end());
  tokens_.clear();
  // The failure and the end are reported once, after every token pushed before them.
  batch.failure = std::exchange(failure_, std::nullopt);
  batch.end = std::exchange(end_, false);
  return batch;
}

void Job::started() {
  std::lock_guard lock(mutex_);
  t_start_ = Clock::now();
}

void Job::set_prefill_stats(model::GenerateStats stats) {
  {
    std::lock_guard lock(mutex_);
    prefill_stats_ = std::move(stats);
  }
  cv_.notify_all();
}

void Job::push(const std::vector<int32_t>& tokens) {
  if (tokens.empty()) return;
  {
    std::lock_guard lock(mutex_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  }
  cv_.notify_all();
}

void Job::fail(std::string message) {
  {
    std::lock_guard lock(mutex_);
    failure_ = std::move(message);
  }
  cv_.notify_all();
}

void Job::finish() {
  {
    std::lock_guard lock(mutex_);
    end_ = true;
    finished_ = true;
  }
  cv_.notify_all();
}

Scheduler::Scheduler(model::Generator& generator, std::vector<int32_t> stop_ids, int max_queue)
    : generator_(generator),
      stop_list_(std::move(stop_ids)),
      stop_ids_(stop_list_.begin(), stop_list_.end()),
      max_queue_(max_queue),
      max_batch_(std::max(1, generator.max_batch_size())),
      thread_([this] {
        if (max_batch_ > 1) loop_batched();
        else loop();
      }) {}

Scheduler::~Scheduler() { close(); }

bool Scheduler::submit(std::shared_ptr<Job> job) {
  {
    std::lock_guard lock(mutex_);
    const int cap = max_batch_ + max_queue_;
    if (closed_ || int(waiting_.size()) + int(active_.size()) >= cap) return false;
    waiting_.push_back(std::move(job));
  }
  cv_.notify_all();
  return true;
}

void Scheduler::withdraw(const std::shared_ptr<Job>& job) {
  job->cancel();
  std::lock_guard lock(mutex_);
  auto it = std::find(waiting_.begin(), waiting_.end(), job);
  if (it != waiting_.end()) waiting_.erase(it);
}

SchedulerStats Scheduler::stats() const {
  std::lock_guard lock(mutex_);
  const int act = int(active_.size());
  return {int(waiting_.size()), act > 0, act, max_batch_, max_queue_};
}

void Scheduler::close() {
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto& job : waiting_) {
      job->cancel();
      job->fail("server shutting down");
      job->finish();
    }
    waiting_.clear();
    for (auto& job : active_) job->cancel();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Scheduler::loop() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || closed_; });
      if (closed_) return;
      job = waiting_.front();
      waiting_.pop_front();
      active_.push_back(job);
    }
    run(*job);
    job->finish();
    std::lock_guard lock(mutex_);
    active_.clear();
  }
}

void Scheduler::loop_batched() {
  struct SlotEntry {
    std::shared_ptr<Job> job;
    int slot = -1;
    int32_t last_token = 0;
    int generated = 0;
    int max_new_tokens = 0;
  };
  struct PrefillEntry {
    std::shared_ptr<Job> job;
    int slot = -1;
    int eff_max = 0;
  };
  std::vector<SlotEntry> live;
  std::vector<PrefillEntry> prefilling;
  live.reserve(std::size_t(max_batch_));
  prefilling.reserve(std::size_t(max_batch_));
  {
    std::lock_guard lock(mutex_);
    active_.reserve(std::size_t(max_batch_));
  }

  // Closed-loop integral time-credit controller for mixed read+decode scheduling.
  // Tracks pure decode step latency ema_step_ms[D] (prompt-independent hardware scaling) so that
  // each active decoder keeps >= (mixed_decode_ratio + margin) of max_batch_ decode step rate.
  double prefill_credit_ms = 0.0;
  double ema_chunk_ms_std = 245.0;    // C_mix ~ 512
  double ema_chunk_ms_small = 155.0;  // C_mix <= 256
  std::vector<double> ema_step_ms(std::size_t(max_batch_ + 1), 0.0);
  std::vector<int> ema_step_n(std::size_t(max_batch_ + 1), 0);

  auto default_rel_speed = [&](int d) -> double {
    if (d >= max_batch_) return 1.0;
    if (max_batch_ == 4) {
      if (d == 1) return 1.60;
      if (d == 2) return 1.38;
      if (d == 3) return 1.15;
    }
    return 1.0 + 0.60 * double(max_batch_ - d) / double(std::max(1, max_batch_ - 1));
  };

  auto remove_active_job = [&](const std::shared_ptr<Job>& job) {
    std::lock_guard lock(mutex_);
    auto it = std::find(active_.begin(), active_.end(), job);
    if (it != active_.end()) active_.erase(it);
  };

  auto retire = [&](SlotEntry& e) {
    if (e.slot >= 0) {
      generator_.release_slot(e.slot);
      e.slot = -1;
    }
    e.job->finish();
    remove_active_job(e.job);
  };

  auto retire_prefill = [&](PrefillEntry& p) {
    if (p.slot >= 0) {
      generator_.release_slot(p.slot);
      p.slot = -1;
    }
    p.job->finish();
    remove_active_job(p.job);
  };

  auto finish_prefill_entry = [&](PrefillEntry& p, int32_t first_tok, const model::GenerateStats& st) {
    p.job->set_prefill_stats(st);
    const int recomputed = std::max(0, st.prompt_tokens - std::clamp(st.reused_tokens, 0, st.prompt_tokens));
    const std::string restored_str =
        st.restored_bytes ? std::to_string(*st.restored_bytes) : std::string("n/a");
    server_log().info("{}: slot={} reuse={} reused={} recomputed={} restored_bytes={} prefill={:.3f}s", p.job->id,
                      p.slot, model::to_string(st.reuse), st.reused_tokens, recomputed, restored_str,
                      st.prefill_seconds);
    if (!p.job->images.empty()) {
      server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", p.job->id, p.job->images.size(),
                        st.image_tokens, st.images_encoded, st.vision_seconds);
    }
    if (p.job->cancelled()) {
      retire_prefill(p);
      return;
    }
    p.job->push({first_tok});
    if (stop_ids_.contains(first_tok) || p.eff_max <= 1 || p.job->cancelled()) {
      retire_prefill(p);
    } else {
      live.push_back(SlotEntry{std::move(p.job), p.slot, first_tok, 1, p.eff_max});
      p.slot = -1;
    }
  };

  auto pick_srtf_prefill = [&]() -> std::size_t {
    std::size_t best = 0;
    int best_rem = generator_.prefill_remaining_tokens(prefilling[0].slot);
    for (std::size_t i = 1; i < prefilling.size(); ++i) {
      const int rem = generator_.prefill_remaining_tokens(prefilling[i].slot);
      if (rem < best_rem) {
        best_rem = rem;
        best = i;
      }
    }
    return best;
  };

  for (;;) {
    // 1. Check shutdown & retire any cancelled active/prefilling slots immediately.
    bool shutting_down = false;
    {
      std::lock_guard lock(mutex_);
      shutting_down = closed_;
    }
    if (shutting_down) {
      for (auto& e : live) retire(e);
      live.clear();
      for (auto& p : prefilling) retire_prefill(p);
      prefilling.clear();
      return;
    }
    for (auto it = live.begin(); it != live.end();) {
      if (it->job->cancelled()) {
        retire(*it);
        it = live.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = prefilling.begin(); it != prefilling.end();) {
      if (it->job->cancelled()) {
        retire_prefill(*it);
        it = prefilling.erase(it);
      } else {
        ++it;
      }
    }

    // 2. Admit newly arrived waiting jobs into free slots (new requests join mid-generation).
    const bool chunked = generator_.supports_chunked_prefill();
    while (int(live.size() + prefilling.size()) < max_batch_) {
      std::shared_ptr<Job> candidate;
      bool need_admit_check = false;
      {
        std::lock_guard lock(mutex_);
        if (closed_ || waiting_.empty()) break;
        const auto& front = waiting_.front();
        if (front->cancelled()) {
          candidate = front;
          waiting_.pop_front();
        } else if (!live.empty() || !prefilling.empty()) {
          candidate = front;
          need_admit_check = true;
        } else {
          candidate = front;
          active_.push_back(candidate);
          waiting_.pop_front();
        }
      }
      if (!need_admit_check && candidate->cancelled()) {
        candidate->finish();
        continue;
      }
      if (need_admit_check) {
        // Perform prompt-aware admission check OUTSIDE mutex_ and inside a per-job try/catch so any
        // exception fails only `candidate` while active live slots and subsequent waiting jobs continue.
        const int cn = int(candidate->prompt_ids.size());
        int c_eff = candidate->max_new_tokens;
        if (!candidate->sampling.forced.empty()) c_eff = std::min(c_eff, int(candidate->sampling.forced.size()));
        const int max_len = generator_.max_model_len();
        if (cn + c_eff > max_len) c_eff = std::max(0, max_len - cn);
        bool admitted = false;
        std::optional<std::string> admit_err;
        try {
          admitted = generator_.can_admit_slot(
              model::Prompt{candidate->prompt_ids, candidate->images, candidate->root_prefix_tokens,
                            candidate->cache_namespace, candidate->bypass_prefix_cache},
              c_eff, candidate->sampling);
        } catch (const std::exception& e) {
          server_log().error("admission check failed: {}: {}", type_name(e), e.what());
          admit_err = "generation failed: " + type_name(e) + ": " + e.what();
        } catch (...) {
          server_log().error("admission check failed: unknown exception");
          admit_err = "generation failed: unknown exception";
        }
        if (admit_err) {
          bool owned = false;
          {
            std::lock_guard lock(mutex_);
            auto it = std::find(waiting_.begin(), waiting_.end(), candidate);
            if (it != waiting_.end()) {
              waiting_.erase(it);
              owned = true;
            }
          }
          // Only fail/finish `candidate` if we still owned it in waiting_; close() or withdraw()
          // may already have finalized it while can_admit_slot() was running.
          if (owned) {
            candidate->fail(*admit_err);
            candidate->finish();
          }
          continue;
        }
        if (!admitted) break;  // wait until an active slot frees KV pages
        {
          std::lock_guard lock(mutex_);
          if (closed_ || waiting_.empty() || waiting_.front() != candidate) continue;
          if (candidate->cancelled()) {
            waiting_.pop_front();
            candidate->finish();
            continue;
          }
          active_.push_back(candidate);
          waiting_.pop_front();
        }
      }
      std::shared_ptr<Job> job = std::move(candidate);
      job->started();
      const int n = int(job->prompt_ids.size());
      int eff_max = job->max_new_tokens;
      if (!job->sampling.forced.empty()) eff_max = std::min(eff_max, int(job->sampling.forced.size()));
      const int max_len = generator_.max_model_len();
      if (n + eff_max > max_len) eff_max = max_len - n;
      try {
        if (eff_max <= 0) {
          throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                      std::to_string(max_len));
        }
        if (chunked) {
          const auto t_beg0 = Clock::now();
          const int slot = generator_.begin_prefill_slot(
              model::Prompt{job->prompt_ids, job->images, job->root_prefix_tokens, job->cache_namespace,
                            job->bypass_prefix_cache},
              eff_max, job->sampling);
          const double dt_beg_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_beg0).count();
          if (!live.empty() && dt_beg_ms > 0.5) prefill_credit_ms -= dt_beg_ms;
          if (slot < 0) {
            std::lock_guard lock(mutex_);
            auto it = std::find(active_.begin(), active_.end(), job);
            if (it != active_.end()) active_.erase(it);
            if (!closed_) waiting_.push_front(job);
            else {
              job->fail("server shutting down");
              job->finish();
            }
            break;
          }
          prefilling.push_back(PrefillEntry{std::move(job), slot, eff_max});
          continue;
        }
        int32_t first_tok = 0;
        model::GenerateStats st;
        const int slot = generator_.start_slot(
            model::Prompt{job->prompt_ids, job->images, job->root_prefix_tokens, job->cache_namespace,
                          job->bypass_prefix_cache},
            eff_max, job->sampling, first_tok, st);
        if (slot < 0) {
          // Put back at front of queue; retry after next step frees resources.
          std::lock_guard lock(mutex_);
          auto it = std::find(active_.begin(), active_.end(), job);
          if (it != active_.end()) active_.erase(it);
          if (!closed_) waiting_.push_front(job);
          else {
            job->fail("server shutting down");
            job->finish();
          }
          break;
        }
        struct SlotGuard {
          model::Generator& gen;
          int slot;
          void release() { slot = -1; }
          ~SlotGuard() {
            if (slot >= 0) gen.release_slot(slot);
          }
        } slot_guard{generator_, slot};

        job->set_prefill_stats(st);
        const int recomputed = std::max(0, st.prompt_tokens - std::clamp(st.reused_tokens, 0, st.prompt_tokens));
        const std::string restored_str =
            st.restored_bytes ? std::to_string(*st.restored_bytes) : std::string("n/a");
        server_log().info("{}: slot={} reuse={} reused={} recomputed={} restored_bytes={} prefill={:.3f}s", job->id,
                          slot, model::to_string(st.reuse), st.reused_tokens, recomputed, restored_str,
                          st.prefill_seconds);
        if (!job->images.empty()) {
          server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", job->id, job->images.size(),
                            st.image_tokens, st.images_encoded, st.vision_seconds);
        }
        if (job->cancelled()) {
          job->finish();
          remove_active_job(job);
          continue;
        }
        job->push({first_tok});
        if (stop_ids_.contains(first_tok) || eff_max <= 1 || job->cancelled()) {
          job->finish();
          remove_active_job(job);
        } else {
          live.push_back(SlotEntry{std::move(job), slot, first_tok, 1, eff_max});
          slot_guard.release();
        }
      } catch (const std::exception& e) {
        server_log().error("generator failed: {}: {}", type_name(e), e.what());
        job->fail("generation failed: " + type_name(e) + ": " + e.what());
        job->finish();
        remove_active_job(job);
      } catch (...) {
        server_log().error("generator failed: unknown exception");
        job->fail("generation failed: unknown exception");
        job->finish();
        remove_active_job(job);
      }
    }

    // 3. Sleep if no active or prefilling sequences and nothing waiting.
    if (live.empty() && prefilling.empty()) {
      prefill_credit_ms = 0.0;
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] { return !waiting_.empty() || closed_; });
      continue;
    }

    // 4a. Pure-prefill fast path: no active decoders, so run full-sized prefill chunks at 100% speed (SRTF).
    if (live.empty()) {
      prefill_credit_ms = 0.0;
      const std::size_t idx = pick_srtf_prefill();
      PrefillEntry& p = prefilling[idx];
      try {
        int32_t first_tok = 0;
        model::GenerateStats st;
        const bool done = generator_.step_prefill_slot(p.slot, 0, first_tok, st);
        if (done) {
          finish_prefill_entry(p, first_tok, st);
          prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
        }
      } catch (const std::exception& e) {
        server_log().error("prefill chunk failed: {}: {}", type_name(e), e.what());
        p.job->fail("generation failed: " + type_name(e) + ": " + e.what());
        retire_prefill(p);
        prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
      } catch (...) {
        server_log().error("prefill chunk failed: unknown exception");
        p.job->fail("generation failed: unknown exception");
        retire_prefill(p);
        prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
      }
      continue;
    }

    // 4b. Execute ONE batched decode step across all live slots and stream tokens immediately.
    const int d_before = std::clamp(int(live.size()), 1, max_batch_);
    std::vector<model::BatchSlotStep> inputs(live.size());
    std::vector<model::BatchSlotOutput> outputs(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) inputs[i] = {live[i].slot, live[i].last_token};

    const auto t_dec0 = Clock::now();
    try {
      generator_.step_slots_multi(inputs, outputs);
    } catch (const std::exception& e) {
      server_log().error("batched step failed: {}: {}", type_name(e), e.what());
      const std::string msg = "generation failed: " + type_name(e) + ": " + e.what();
      for (auto& entry : live) {
        entry.job->fail(msg);
        retire(entry);
      }
      live.clear();
      continue;
    } catch (...) {
      server_log().error("batched step failed: unknown exception");
      for (auto& entry : live) {
        entry.job->fail("generation failed: unknown exception");
        retire(entry);
      }
      live.clear();
      continue;
    }
    const double dt_dec_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_dec0).count();

    int toks_emitted = 0;
    std::vector<SlotEntry> kept;
    kept.reserve(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) {
      SlotEntry& e = live[i];
      if (e.job->cancelled()) {
        retire(e);
        continue;
      }
      bool done = false;
      for (int k = 0; k < outputs[i].count; ++k) {
        const int32_t tok = outputs[i].tokens[k];
        e.job->push({tok});
        e.last_token = tok;
        ++e.generated;
        ++toks_emitted;
        if (stop_ids_.contains(tok) || e.generated >= e.max_new_tokens || e.job->cancelled()) {
          done = true;
          break;
        }
      }
      if (done) {
        retire(e);
      } else {
        kept.push_back(std::move(e));
      }
    }
    live = std::move(kept);

    if (dt_dec_ms > 0.5 && toks_emitted > 0) {
      if (ema_step_n[d_before] == 0) ema_step_ms[d_before] = dt_dec_ms;
      else ema_step_ms[d_before] = 0.92 * ema_step_ms[d_before] + 0.08 * dt_dec_ms;
      ema_step_n[d_before] = std::min(ema_step_n[d_before] + 1, 1000000);
    }

    // 5. Mixed read+decode closed-loop prefill budget: run one bounded prefill slice only when
    //    accumulated time credit guarantees active decoders stay above mixed_decode_ratio * max_batch speed.
    if (prefilling.empty()) {
      prefill_credit_ms = 0.0;
    } else if (!live.empty()) {
      const int d_now = std::clamp(int(live.size()), 1, max_batch_);
      double rel = default_rel_speed(d_now);
      if (ema_step_n[d_now] >= 12 && ema_step_n[max_batch_] >= 12 && ema_step_ms[d_now] > 1.0) {
        const double measured_rel = std::clamp(ema_step_ms[max_batch_] / ema_step_ms[d_now], 1.0, 2.5);
        // Take the conservative minimum against the prior ceiling so single-slot variance never starves decode.
        rel = std::min(rel * 1.10, measured_rel);
      }
      const double target = std::clamp(double(generator_.mixed_decode_ratio()) + 0.05, 0.50, 0.95);
      const double min_dec_share = std::clamp(target / std::max(1.0, rel), 0.45, 0.92);
      const double prefill_ratio = (1.0 - min_dec_share) / min_dec_share;

      const int base_chunk = std::max(64, generator_.prefill_interleave_chunk());
      const int c_mix = (d_now >= 3) ? std::min(base_chunk, 256) : base_chunk;
      double& ema_chunk = (c_mix <= 256) ? ema_chunk_ms_small : ema_chunk_ms_std;

      prefill_credit_ms = std::min(prefill_credit_ms + dt_dec_ms * prefill_ratio, 1.5 * ema_chunk);

      const std::size_t idx = pick_srtf_prefill();
      const int rem = generator_.prefill_remaining_tokens(prefilling[idx].slot);
      const double frac = std::clamp(double(std::max(1, rem)) / double(c_mix), 0.15, 1.0);
      const double trigger_ms = ema_chunk * frac;

      if (prefill_credit_ms >= trigger_ms) {
        PrefillEntry& p = prefilling[idx];
        const auto t_pre0 = Clock::now();
        try {
          int32_t first_tok = 0;
          model::GenerateStats st;
          const bool done = generator_.step_prefill_slot(p.slot, c_mix, first_tok, st);
          const double dt_pre_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_pre0).count();
          if (rem >= c_mix && dt_pre_ms > 1.0) {
            ema_chunk = 0.80 * ema_chunk + 0.20 * dt_pre_ms;
          }
          prefill_credit_ms -= dt_pre_ms;
          if (done) {
            finish_prefill_entry(p, first_tok, st);
            prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
          }
        } catch (const std::exception& e) {
          server_log().error("prefill chunk failed: {}: {}", type_name(e), e.what());
          p.job->fail("generation failed: " + type_name(e) + ": " + e.what());
          retire_prefill(p);
          prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
        } catch (...) {
          server_log().error("prefill chunk failed: unknown exception");
          p.job->fail("generation failed: unknown exception");
          retire_prefill(p);
          prefilling.erase(prefilling.begin() + std::ptrdiff_t(idx));
        }
      }
    }
  }
}

void Scheduler::run(Job& job) {
  if (job.cancelled()) return;
  job.started();
  int n = 0;
  std::vector<int32_t> out;
  const auto on_tokens = [&](std::span<const int32_t> tokens) -> bool {
    out.clear();
    bool more = true;
    for (int32_t tok : tokens) {
      if (job.cancelled()) {
        more = false;
        break;
      }
      out.push_back(tok);
      ++n;
      if (stop_ids_.contains(tok) || n >= job.max_new_tokens || job.cancelled()) {
        more = false;
        break;
      }
    }
    job.push(out);
    return more;
  };
  try {
    const auto on_prefill = [&](const model::GenerateStats& pst) { job.set_prefill_stats(pst); };
    const model::GenerateStats st = generator_.generate(
        model::Prompt{job.prompt_ids, job.images, job.root_prefix_tokens, job.cache_namespace,
                      job.bypass_prefix_cache},
        job.max_new_tokens, job.sampling, stop_list_, on_tokens, on_prefill);
    job.set_prefill_stats(st);
    // prefix reuse of the prompt: the path, the tokens reused and recomputed, restored bytes, prefill time
    const int recomputed = std::max(0, st.prompt_tokens - std::clamp(st.reused_tokens, 0, st.prompt_tokens));
    const std::string restored_str =
        st.restored_bytes ? std::to_string(*st.restored_bytes) : std::string("n/a");
    server_log().info("{}: reuse={} reused={} recomputed={} restored_bytes={} prefill={:.3f}s", job.id,
                      model::to_string(st.reuse), st.reused_tokens, recomputed, restored_str,
                      st.prefill_seconds);
    if (!job.images.empty()) {  // images: all of the prompt, those the vision tower encoded now, its time
      server_log().info("{}: images={} image_tokens={} encoded={} vit={:.3f}s", job.id, job.images.size(),
                        st.image_tokens, st.images_encoded, st.vision_seconds);
    }
  } catch (const std::exception& e) {  // report to the client instead of killing the worker
    server_log().error("generator failed: {}: {}", type_name(e), e.what());
    job.fail("generation failed: " + type_name(e) + ": " + e.what());
  } catch (...) {
    server_log().error("generator failed: unknown exception");
    job.fail("generation failed: unknown exception");
  }
}

}  // namespace ingine::engine
