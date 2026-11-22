// sokoban-solver: a push-optimal Sokoban solver built for make.fun's generated levels.
//
// The levels are a hallway with independent rooms off it: a box can never leave its room,
// so a level's push optimum is the sum of its rooms' optima. This solver therefore
//   1. splits the level into rooms (the hallway's dead squares are cut away; each part of
//      the floor left that holds boxes is one room),
//   2. solves every room push-optimally with an A* search over pushes, guided by a
//      minimum-cost box-to-goal matching over precomputed single-box push distances,
//      with dead-square and frozen-block deadlock pruning,
//   3. joins the rooms' move strings, the player walking on from room to room.
// Every budget is a count of search expansions, never the clock, so two runs of the
// program on the same set write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Per-room A* expansion budget (work, not time); the greedy fallback gets the same again. */
const ASTAR_BUDGET = 1_000_000;
const GREEDY_BUDGET = 1_000_000;

/** The four directions: [dx, dy, walk letter, push letter]. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: walls, goals, boxes and the player, short rows padded with walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/** Flood the squares the player reaches from `from` around walls and boxes; returns the fill and its lowest square. */
function flood(w: number, wall: Uint8Array, boxAt: Uint8Array, from: number): { seen: Uint8Array; low: number } {
  const seen = new Uint8Array(wall.length);
  const stack = [from];
  seen[from] = 1;
  let low = from;
  while (stack.length > 0) {
    const p = stack.pop()!;
    if (p < low) low = p;
    const up = p - w;
    const dn = p + w;
    const lf = p - 1;
    const rt = p + 1;
    if (!wall[up] && !boxAt[up] && !seen[up]) { seen[up] = 1; stack.push(up); }
    if (!wall[dn] && !boxAt[dn] && !seen[dn]) { seen[dn] = 1; stack.push(dn); }
    if (!wall[lf] && !boxAt[lf] && !seen[lf]) { seen[lf] = 1; stack.push(lf); }
    if (!wall[rt] && !boxAt[rt] && !seen[rt]) { seen[rt] = 1; stack.push(rt); }
  }
  return { seen, low };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower-case LURD (first direction wins ties). */
function walkPath(w: number, wall: Uint8Array, boxAt: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || boxAt[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Single-box push distances on a board: dist[goalIndex * size + cell] is the fewest pushes that move one box
 * from `cell` onto that goal, ignoring the other boxes (the player may stand on any non-wall square the push
 * needs). Infinity (a large sentinel) marks a dead square: a box there reaches no goal at all.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array {
  const size = wall.length;
  const INF = 1 << 28;
  const dist = new Int32Array(goals.length * size).fill(INF);
  goals.forEach((g, gi) => {
    const base = gi * size;
    dist[base + g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const y = queue[qi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const x = y - d; // the box stood at x and was pushed in direction d onto y
        if (wall[x] || wall[x - d] || dist[base + x] !== INF) continue;
        dist[base + x] = dist[base + y] + 1;
        queue.push(x);
      }
    }
  });
  return dist;
}

/** Minimum-cost matching of boxes to goals over the push distances (exact for few boxes, a sum of minima beyond). */
function heuristic(dist: Int32Array, size: number, goals: number[], boxes: number[]): number {
  const n = boxes.length;
  const INF = 1 << 28;
  if (n === 0) return 0;
  if (n > 7) {
    let sum = 0;
    for (const b of boxes) {
      let best = INF;
      for (let gi = 0; gi < goals.length; gi++) best = Math.min(best, dist[gi * size + b]);
      if (best >= INF) return INF;
      sum += best;
    }
    return sum;
  }
  // DP over goal subsets: dp[mask] = the best cost of placing the first popcount(mask) boxes.
  const full = 1 << goals.length;
  const dp = new Int32Array(full).fill(INF);
  dp[0] = 0;
  for (let mask = 0; mask < full; mask++) {
    const k = popcount(mask);
    if (k >= n || dp[mask] >= INF) continue;
    const b = boxes[k];
    for (let gi = 0; gi < goals.length; gi++) {
      if (mask & (1 << gi)) continue;
      const c = dist[gi * size + b];
      if (c >= INF) continue;
      const next = mask | (1 << gi);
      if (dp[mask] + c < dp[next]) dp[next] = dp[mask] + c;
    }
  }
  return dp[full - 1];
}

function popcount(x: number): number {
  let c = 0;
  while (x) { x &= x - 1; c++; }
  return c;
}

/** Whether the box just pushed onto `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, boxAt: Uint8Array, goalAt: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || boxAt[a]) || !(wall[b] || boxAt[b]) || !(wall[c] || boxAt[c]) || !(wall[d] || boxAt[d])) continue;
    if ((boxAt[a] && !goalAt[a]) || (boxAt[b] && !goalAt[b]) || (boxAt[c] && !goalAt[c]) || (boxAt[d] && !goalAt[d])) return true;
  }
  return false;
}

interface SubProblem {
  w: number;
  wall: Uint8Array; // the board with every other room's boxes standing as walls
  goals: number[];
  boxes: number[];
  player: number;
}

/**
 * A push-level search over one room: A* (optimal) when `optimal`, greedy best-first otherwise.
 * Returns the move string that solves the room, or null when the expansion budget runs out.
 */
function searchRoom(sub: SubProblem, budget: number, optimal: boolean): string | null {
  const { w, wall, goals } = sub;
  const size = wall.length;
  const goalAt = new Uint8Array(size);
  for (const g of goals) goalAt[g] = 1;
  const dist = pushDistances(w, wall, goals);
  const INF = 1 << 28;
  const dead = new Uint8Array(size);
  for (let cell = 0; cell < size; cell++) {
    if (wall[cell]) continue;
    let alive = false;
    for (let gi = 0; gi < goals.length; gi++) if (dist[gi * size + cell] < INF) { alive = true; break; }
    if (!alive) dead[cell] = 1;
  }
  if (sub.boxes.some((b) => dead[b])) return null;

  const boxAt = new Uint8Array(size);
  const keyOf = (boxes: number[], low: number): string => String.fromCharCode(low, ...boxes);

  // Node store: parallel arrays; a node is its index.
  const nBoxes: number[][] = [sub.boxes];
  const nPlayer: number[] = [sub.player]; // a square of the node's player region (where the last push left the player)
  const nG: number[] = [0];
  const nParent: number[] = [-1];
  const nFrom: number[] = [-1]; // the pushed box's square before the push
  const nTo: number[] = [-1]; // ... and after it
  const best = new Map<string, number>();

  const startBoxAt = new Uint8Array(size);
  for (const b of sub.boxes) startBoxAt[b] = 1;
  const startRegion = flood(w, wall, startBoxAt, sub.player);
  best.set(keyOf(sub.boxes, startRegion.low), 0);

  // Binary heap of node indices on (priority, insertion order): fully deterministic.
  const heap: number[] = [0];
  const prio: number[] = [heuristic(dist, size, goals, sub.boxes)];
  const order: number[] = [0];
  let counter = 1;
  const less = (a: number, b: number): boolean => prio[a] < prio[b] || (prio[a] === prio[b] && order[a] < order[b]);
  const heapPush = (node: number, p: number): void => {
    prio[node] = p;
    order[node] = counter++;
    let i = heap.length;
    heap.push(node);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!less(heap[i], heap[parent])) break;
      [heap[i], heap[parent]] = [heap[parent], heap[i]];
      i = parent;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  let found = sub.boxes.every((b) => goalAt[b]) ? 0 : -1;
  let expanded = 0;
  while (heap.length > 0 && found < 0) {
    const node = heapPop();
    const boxes = nBoxes[node];
    const g = nG[node];
    for (const b of boxes) boxAt[b] = 1;
    const region = flood(w, wall, boxAt, nPlayer[node]);
    // A stale heap entry: this state was since reached with fewer pushes.
    const known = best.get(keyOf(boxes, region.low));
    if (known !== undefined && known < g) {
      for (const b of boxes) boxAt[b] = 0;
      continue;
    }
    if (expanded >= budget) {
      for (const b of boxes) boxAt[b] = 0;
      break;
    }
    expanded++;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const stand = b - d;
        const target = b + d;
        if (!region.seen[stand] || wall[target] || boxAt[target] || dead[target]) continue;
        boxAt[b] = 0;
        boxAt[target] = 1;
        const frozen = frozenBlock(w, wall, boxAt, goalAt, target);
        let child = -1;
        if (!frozen) {
          const nextBoxes = boxes.slice();
          nextBoxes[bi] = target;
          nextBoxes.sort((x, y) => x - y);
          const childRegion = flood(w, wall, boxAt, b);
          const key = keyOf(nextBoxes, childRegion.low);
          const ng = g + 1;
          const seenG = best.get(key);
          if (seenG === undefined || ng < seenG) {
            const h = heuristic(dist, size, goals, nextBoxes);
            if (h < INF) {
              best.set(key, ng);
              child = nBoxes.length;
              nBoxes.push(nextBoxes);
              nPlayer.push(b); // the push leaves the player on the box's old square
              nG.push(ng);
              nParent.push(node);
              nFrom.push(b);
              nTo.push(target);
              heapPush(child, optimal ? ng + h : h * 4 + ng);
              if (nextBoxes.every((x) => goalAt[x])) found = child;
            }
          }
        }
        boxAt[target] = 0;
        boxAt[b] = 1;
      }
    }
    for (const b of boxes) boxAt[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push's stand square, then push.
  const chain: number[] = [];
  for (let s = found; s > 0; s = nParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = sub.player;
  const curBoxes = sub.boxes.slice();
  const curBoxAt = new Uint8Array(size);
  for (const b of curBoxes) curBoxAt[b] = 1;
  for (const s of chain) {
    const from = nFrom[s];
    const to = nTo[s];
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    moves += walkPath(w, wall, curBoxAt, player, from - d);
    moves += dir[3];
    curBoxAt[from] = 0;
    curBoxAt[to] = 1;
    curBoxes[curBoxes.indexOf(from)] = to;
    player = from;
  }
  return moves;
}

/** Solve one room optimally, falling back to a greedy search that still solves it within budget. */
function solveRoom(sub: SubProblem): string | null {
  return searchRoom(sub, ASTAR_BUDGET, true) ?? searchRoom(sub, GREEDY_BUDGET, false);
}

/** The squares a box alone can be pushed onto some goal from (pulled back from the goals over open floor). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  while (queue.length > 0) {
    const x = queue.shift()!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal)
 * out of the floor; each part left that holds boxes is a room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const parts: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = parts.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = parts.length;
          squares.push(q);
        }
      }
    }
    parts.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) parts[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    parts[part[g]].goals.push(g);
  }
  if (parts.some((p) => p.boxes.length !== p.goals.length)) return null;
  return parts.filter((p) => p.boxes.length > 0);
}

/** Where the player stands after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Solve a whole level: room by room when it decomposes, as one search when it does not. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solveRoom({ w: level.w, wall: level.wall, goals: level.goals, boxes: level.boxes, player: level.player });
    return whole ?? '';
  }
  let player = level.player;
  let moves = '';
  const standing = new Map<Room, number[]>(); // each room's boxes as they stand now (its goals once solved)
  for (const room of rooms) standing.set(room, room.boxes.slice());
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of standing.get(other)!) wall[b] = 1;
    }
    const solved = solveRoom({ w: level.w, wall, goals: room.goals.slice().sort((a, b) => a - b), boxes: room.boxes.slice().sort((a, b) => a - b), player });
    if (solved === null) return '';
    moves += solved;
    player = endOf(level.w, player, solved);
    standing.set(room, room.goals.slice());
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
