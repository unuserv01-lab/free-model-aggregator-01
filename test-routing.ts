// Deterministic routing tests for the provider-aware tryOrder (lib/providers.ts).
// Stdlib only. Run: node test-routing.ts  — must print PASS for every case and exit 0.
// The tests import tryOrder / penalizedRating / noteCall / isExhaustingStatus from
// lib/providers.ts directly; they do not re-implement the routing logic.
import { strict as assert } from 'node:assert';
import {
  FAILURE_PENALTY_MILLI,
  isExhaustingStatus,
  noteCall,
  penalizedRating,
  tryOrder,
  type Candidate,
  type FailureStreaks,
  type Probe,
  type ProbedModel,
} from './lib/providers.ts';

function probe(id: string): Probe {
  return {
    provider: { id, name: id, wire: 'openai', auth: 'bearer', base: `/p/${id}`, key_env: 'K', key: 'k' },
    ok: true,
    status: 200,
    models: [],
    error: null,
  };
}

function cand(p: Probe, modelId: string, rating: number): Candidate {
  const model: ProbedModel = { id: modelId, display_name: modelId, free: true, context_length: null, tools: null, vision: null, json: null };
  return { probe: p, model, row: null, rating };
}

function ids(order: readonly Candidate[]): string[] {
  return order.map((c) => `${c.probe.provider.id}/${c.model.id}`);
}

/** Walk an order the way serve() does: skip providers exhausted by a 402/429/0 failure. */
function simulate(order: readonly Candidate[], fail: (c: Candidate) => number | null): string[] {
  const exhausted = new Set<string>();
  const attempted: string[] = [];
  for (const c of order) {
    if (exhausted.has(c.probe.provider.id)) continue;
    attempted.push(`${c.probe.provider.id}/${c.model.id}`);
    const status = fail(c);
    if (status === null) break;
    if (isExhaustingStatus(status)) exhausted.add(c.probe.provider.id);
  }
  return attempted;
}

const streaks = (): FailureStreaks => new Map();

// Case 1: provider ranking uses the penalized rating. A best 900 with streak 4 loses
// 4 * 6 = 24 milli, so B best 880 (streak 0) ranks first.
{
  const a = probe('a');
  const b = probe('b');
  const s = streaks();
  noteCall(s, 'a', false);
  noteCall(s, 'a', false);
  noteCall(s, 'a', false);
  noteCall(s, 'a', false);
  assert.equal(penalizedRating(cand(a, 'a1', 900), s), 900 - 4 * FAILURE_PENALTY_MILLI);
  const order = tryOrder([cand(a, 'a1', 900), cand(b, 'b1', 880)], s);
  assert.deepEqual(ids(order), ['b/b1', 'a/a1']);
  console.log('PASS 1: penalized provider ranking (streak drops A below B)');
}

// Case 2: candidates are grouped by provider, best model first inside each provider.
{
  const a = probe('a');
  const b = probe('b');
  const c = probe('c');
  const order = tryOrder([cand(b, 'b1', 800), cand(a, 'a2', 850), cand(c, 'c1', 700), cand(a, 'a1', 900)], streaks());
  assert.deepEqual(ids(order), ['a/a1', 'a/a2', 'b/b1', 'c/c1']);
  console.log('PASS 2: provider-grouped order');
}

// Case 3: Provider A 429, Provider B succeeds — the rest of A's models are skipped.
{
  const a = probe('a');
  const b = probe('b');
  const order = tryOrder([cand(a, 'a1', 900), cand(a, 'a2', 850), cand(b, 'b1', 800)], streaks());
  const attempted = simulate(order, (c) => (c.probe.provider.id === 'a' ? 429 : null));
  assert.deepEqual(attempted, ['a/a1', 'b/b1']);
  console.log('PASS 3: A 429 exhausts A, B serves, A2 never tried');
}

// Case 4: Provider A has several models and they all fail with 500 (not exhausting) —
// the chain still walks A's models, then reaches B.
{
  const a = probe('a');
  const b = probe('b');
  const order = tryOrder([cand(a, 'a1', 900), cand(a, 'a2', 850), cand(b, 'b1', 800)], streaks());
  const attempted = simulate(order, (c) => (c.probe.provider.id === 'a' ? 500 : null));
  assert.deepEqual(attempted, ['a/a1', 'a/a2', 'b/b1']);
  assert.equal(isExhaustingStatus(500), false);
  assert.equal(isExhaustingStatus(402), true);
  assert.equal(isExhaustingStatus(429), true);
  assert.equal(isExhaustingStatus(0), true);
  console.log('PASS 4: A fails on 500s, B still reachable; 402/429/0 exhaust, 500 does not');
}

// Case 5: all providers fail — the walk terminates, bounded by the candidate count.
{
  const a = probe('a');
  const b = probe('b');
  const order = tryOrder([cand(a, 'a1', 900), cand(b, 'b1', 800)], streaks());
  const attempted = simulate(order, () => 429);
  assert.deepEqual(attempted, ['a/a1', 'b/b1']);
  console.log('PASS 5: all providers fail, walk ends without hanging');
}

// Case 6: replay determinism — the same input gives the same order, twice.
{
  const a = probe('a');
  const b = probe('b');
  const s = streaks();
  noteCall(s, 'b', false);
  const mk = () => [cand(a, 'a1', 900), cand(b, 'b1', 880), cand(a, 'a2', 850)];
  assert.deepEqual(ids(tryOrder(mk(), s)), ids(tryOrder(mk(), s)));
  console.log('PASS 6: deterministic replay order');
}

console.log('ALL ROUTING TESTS PASS');
