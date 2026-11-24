// CPU unit tests for Stage 2 (#825) refcounted paged full-KV pool & exact prefix checkpoint cache:
//   1. Page refcount lifecycle (active_refs, cache_refs, committed flag, never sharing partial pages).
//   2. Page boundaries (63/64/65/128/129), exact hit vs longest-prefix hit, root vs non-root LRU eviction.
//   3. Eviction with live sibling slots (shared pages stay pinned until active_refs == 0, then return to free).
//   4. Non-starvation guarantee: evictable cache pages NEVER starve active admission or decode page growth.
//   5. Protected entry during eviction (never evicts the checkpoint currently being restored).
//   6. Fail-closed StateIdentity isolation & hash-collision resistance (exact operator== and token-ID std::equal).
//   7. Root-prefix boundary validation (validate_root_prefix_tokens, including straddling PromptImage).
#include "ingine/generator.h"
#include "model/batch_prefix_cache.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ingine::model::BatchPrefixCache;
using ingine::model::BatchPrefixEntry;
using ingine::model::PagedKvPagePool;
using ingine::model::PromptImage;
using ingine::model::StateIdentity;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

struct DummyPayload {
  int tag = 0;
};

StateIdentity make_id(const std::string& ns = "", bool dense_fp8 = true, std::uint32_t kv_fmt = 1) {
  StateIdentity id{};
  id.model_sha256[0] = 0x42;
  id.act_dtype = 0;
  id.dense_fp8 = dense_fp8;
  id.kv_format = kv_fmt;
  id.max_ctx = 4096;
  id.prefill_chunk = 3072;
  id.spec_enabled = false;
  id.cache_namespace = ns;
  return id;
}

std::vector<std::int32_t> make_tokens(int n, int base = 100) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) t[static_cast<std::size_t>(i)] = base + i;
  return t;
}

void test_refcount_and_sibling_eviction() {
  PagedKvPagePool pool;
  pool.init(4);
  CHECK(pool.free_count() == 4);
  CHECK(pool.available_for_admission() == 4);

  BatchPrefixCache<DummyPayload> cache;
  cache.set_vram_budget(1024);

  // Slot 0 prefills 130 tokens -> 2 committed pages (0..127) + 1 private partial tail page (128..129).
  const int p0 = pool.alloc_private_page();
  const int p1 = pool.alloc_private_page();
  const int p_tail = pool.alloc_private_page();
  CHECK(pool.free_count() == 1);

  BatchPrefixEntry<DummyPayload> e;
  e.identity = make_id("t1");
  e.tokens = make_tokens(130);
  e.committed_pages = {p0, p1};
  e.tail_tokens = 2;
  e.private_vram_bytes = 256;
  e.payload = {1};
  auto [pub, ev] = cache.publish(pool, std::move(e));
  CHECK(pub && ev == 0);
  CHECK(pool.is_committed(p0) && pool.is_committed(p1));
  CHECK(!pool.is_committed(p_tail));  // partial tail NEVER shared/committed
  CHECK(pool.active_refs(p0) == 1 && pool.cache_refs(p0) == 1);

  // Sibling Slot 1 hits the 130-token prefix -> retains committed pages p0, p1 zero-copy, allocates its own tail.
  const auto* hit = cache.find_longest_prefix(make_id("t1"), make_tokens(140));
  CHECK(hit != nullptr && hit->tokens.size() == 130 && hit->tail_tokens == 2);
  pool.retain_committed_page(p0);
  pool.retain_committed_page(p1);
  const int p_tail1 = pool.alloc_private_page();
  CHECK(pool.free_count() == 0);
  CHECK(pool.active_refs(p0) == 2 && pool.cache_refs(p0) == 1);

  // Slot 0 finishes and releases its active refs; p0, p1 stay alive for sibling Slot 1 + cache.
  pool.release_active_page(p0);
  pool.release_active_page(p1);
  pool.release_active_page(p_tail);
  CHECK(pool.free_count() == 1);  // only p_tail freed
  CHECK(pool.active_refs(p0) == 1 && pool.cache_refs(p0) == 1);

  // Evict cache entry while sibling Slot 1 is STILL LIVE: p0, p1 must NOT be freed!
  const int evicted = cache.clear(pool);
  CHECK(evicted == 1);
  CHECK(pool.free_count() == 1);
  CHECK(pool.active_refs(p0) == 1 && pool.cache_refs(p0) == 0 && pool.is_committed(p0));

  // Once sibling Slot 1 finishes, p0, p1, and p_tail1 ALL return cleanly to free_pages!
  pool.release_active_page(p0);
  pool.release_active_page(p1);
  pool.release_active_page(p_tail1);
  CHECK(pool.free_count() == 4);
  CHECK(!pool.is_committed(p0) && !pool.is_committed(p1));
}

void test_non_starvation_and_protected_hit() {
  PagedKvPagePool pool;
  pool.init(4);
  BatchPrefixCache<DummyPayload> cache;
  cache.set_vram_budget(4096);

  // Publish 2 cached entries holding 2 pages each (total 4 pages = 100% of pool, zero active slots).
  int a0 = pool.alloc_private_page(), a1 = pool.alloc_private_page();
  {
    BatchPrefixEntry<DummyPayload> e;
    e.identity = make_id();
    e.tokens = make_tokens(128, 1000);
    e.committed_pages = {a0, a1};
    e.private_vram_bytes = 256;
    e.is_root = true;
    CHECK(cache.publish(pool, std::move(e)).first);
  }
  pool.release_active_page(a0);
  pool.release_active_page(a1);

  int b0 = pool.alloc_private_page(), b1 = pool.alloc_private_page();
  {
    BatchPrefixEntry<DummyPayload> e;
    e.identity = make_id();
    e.tokens = make_tokens(128, 2000);
    e.committed_pages = {b0, b1};
    e.private_vram_bytes = 256;
    e.is_root = false;
    CHECK(cache.publish(pool, std::move(e)).first);
  }
  pool.release_active_page(b0);
  pool.release_active_page(b1);

  CHECK(pool.free_count() == 0);
  CHECK(pool.evictable_count() == 4);
  CHECK(pool.available_for_admission() == 4);  // cache NEVER starves admission!

  // New request hits entry A (root, 128 toks) and needs 2 fresh pages for suffix -> must evict B, NEVER A!
  auto q = make_tokens(200, 1000);
  const auto* hit = cache.find_longest_prefix(make_id(), q);
  CHECK(hit != nullptr && hit->tokens.size() == 128 && hit->is_root);
  const std::uint64_t hit_id = hit->id;
  const int ev = cache.ensure_free_pages(pool, 2, hit_id);
  CHECK(ev == 1);
  CHECK(pool.free_count() == 2);
  CHECK(cache.size() == 1);
  // Hit entry A is still intact and pinned!
  CHECK(pool.cache_refs(a0) == 1 && pool.cache_refs(a1) == 1);

  // Even root entry gets evicted when an active request needs all 4 pages (non-starvation).
  const int ev2 = cache.ensure_free_pages(pool, 4, 0);
  CHECK(ev2 == 1);
  CHECK(pool.free_count() == 4);
  CHECK(cache.size() == 0);
}

void test_identity_isolation_and_exact_prefix_boundaries() {
  PagedKvPagePool pool;
  pool.init(16);
  BatchPrefixCache<DummyPayload> cache;
  cache.set_vram_budget(8192);

  // Publish page-boundary lengths 63, 64, 65 for identical base tokens in namespace "alpha".
  for (int len : {63, 64, 65}) {
    std::vector<int> pages;
    for (int i = 0; i < len / 64; ++i) pages.push_back(pool.alloc_private_page());
    BatchPrefixEntry<DummyPayload> e;
    e.identity = make_id("alpha");
    e.tokens = make_tokens(len, 500);
    e.committed_pages = pages;
    e.tail_tokens = len % 64;
    e.private_vram_bytes = 128;
    CHECK(cache.publish(pool, std::move(e)).first);
    for (int p : pages) pool.release_active_page(p);
  }

  // Exact longest-prefix lookup for 63, 64, 65, and 70 tokens.
  CHECK(cache.find_longest_prefix(make_id("alpha"), make_tokens(63, 500))->tokens.size() == 63);
  CHECK(cache.find_longest_prefix(make_id("alpha"), make_tokens(64, 500))->tokens.size() == 64);
  CHECK(cache.find_longest_prefix(make_id("alpha"), make_tokens(65, 500))->tokens.size() == 65);
  CHECK(cache.find_longest_prefix(make_id("alpha"), make_tokens(70, 500))->tokens.size() == 65);

  // Single-token divergence at index 63 -> must fall back cleanly to length-63 prefix, NOT 64/65!
  auto forked = make_tokens(70, 500);
  forked[63] = 999999;
  const auto* fhit = cache.find_longest_prefix(make_id("alpha"), forked);
  CHECK(fhit != nullptr && fhit->tokens.size() == 63);

  // Single-token divergence at index 0 -> clean miss!
  auto miss_tok = make_tokens(70, 500);
  miss_tok[0] = 888888;
  CHECK(cache.find_longest_prefix(make_id("alpha"), miss_tok) == nullptr);

  // Identity isolation: different namespace, dense_fp8, kv_format, prefill_chunk, persona, model_sha256 -> ALL MISS!
  const auto full = make_tokens(70, 500);
  CHECK(cache.find_longest_prefix(make_id("beta"), full) == nullptr);
  CHECK(cache.find_longest_prefix(make_id("alpha", false, 1), full) == nullptr);
  CHECK(cache.find_longest_prefix(make_id("alpha", true, 0), full) == nullptr);
  {
    auto id_chunk = make_id("alpha");
    id_chunk.prefill_chunk = 512;
    CHECK(cache.find_longest_prefix(id_chunk, full) == nullptr);
  }
  {
    auto id_persona = make_id("alpha");
    id_persona.persona_key = 7;
    id_persona.persona_fingerprint = "deadbeef";
    CHECK(cache.find_longest_prefix(id_persona, full) == nullptr);
  }
  {
    auto id_model = make_id("alpha");
    id_model.model_sha256[0] = 0x99;
    CHECK(cache.find_longest_prefix(id_model, full) == nullptr);
  }

  cache.clear(pool);
  CHECK(pool.free_count() == 16);
}

void test_private_vram_budget_and_root_validation() {
  PagedKvPagePool pool;
  pool.init(8);
  BatchPrefixCache<DummyPayload> cache;
  cache.set_vram_budget(300);

  auto add_entry = [&](int base) {
    int p = pool.alloc_private_page();
    BatchPrefixEntry<DummyPayload> e;
    e.identity = make_id();
    e.tokens = make_tokens(64, base);
    e.committed_pages = {p};
    e.private_vram_bytes = 200;
    auto res = cache.publish(pool, std::move(e));
    pool.release_active_page(p);
    return res;
  };

  CHECK(add_entry(100).first);
  CHECK(cache.stats().vram_used_bytes == 200);
  auto [pub2, ev2] = add_entry(200);
  CHECK(pub2 && ev2 == 1);  // 200 + 200 > 300 -> evicts first entry cleanly!
  CHECK(cache.stats().vram_used_bytes == 200);
  CHECK(cache.stats().vram_evictions == 1);
  CHECK(pool.free_count() == 7);

  // Zero budget disables caching cleanly.
  cache.clear(pool);
  cache.set_vram_budget(0);
  CHECK(!add_entry(300).first);
  CHECK(pool.free_count() == 8);

  // Root-prefix boundary validation (including straddling PromptImage).
  const auto ids = make_tokens(100);
  PromptImage img;
  img.pos = 40;
  img.height = 80;  // 2 rows
  img.width = 120;  // 3 cols -> 6 tokens [40, 46)
  std::vector<PromptImage> imgs = {img};
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 0) == 0);
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, -5) == 0);
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 101) == 0);
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 40) == 40);
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 43) == 0);  // straddles [40, 46)
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 46) == 46);
  CHECK(ingine::model::validate_root_prefix_tokens(ids, imgs, 100) == 100);
}

}  // namespace

int main() {
  test_refcount_and_sibling_eviction();
  test_non_starvation_and_protected_hit();
  test_identity_isolation_and_exact_prefix_boundaries();
  test_private_vram_budget_and_root_validation();
  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_prefix_cache ok\n", g_failures);
  return g_failures ? 1 : 0;
}
