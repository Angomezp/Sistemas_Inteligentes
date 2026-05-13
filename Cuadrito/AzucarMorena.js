class AzucarMorena extends Agent {
  constructor() {
    super();
    this.boardUtil = new Board();
    this.dirs = [[-1, 0, 2], [0, 1, 3], [1, 0, 0], [0, -1, 1]];
    this.bits = new Uint8Array(16);
    for (let i = 0; i < 16; i++) {
      let x = i, c = 0;
      while (x) { x &= x - 1; c++; }
      this.bits[i] = c;
    }
    this.remainingFromCell = new Uint8Array(16);
    for (let i = 0; i < 16; i++) this.remainingFromCell[i] = 4 - this.bits[i];
    
    this.transTable = null;
    this.killerMoves = null;
    this.previousBestMove = null;
    
    this.startTime = 0;
    this.timeLimit = 0;
    this.size = 0;
    this.totalMoves = 0;
    this.initialTime = 0;
  }

  init(color, board, time = 20000) {
    super.init(color, board, time);
    this.size = board.length;
    this.totalMoves = 2 * this.size * (this.size + 1);
    this.initialTime = time;
    this.transTable = new Map();
    this.killerMoves = new Array(64).fill().map(() => []);
    this.previousBestMove = null;
  }

  count(v) { return this.bits[v & 15]; }
  
  clone(board) {
    let n = board.length, b = new Array(n);
    for (let i = 0; i < n; i++) b[i] = board[i].slice();
    return b;
  }
  
  outOfTime() { return Date.now() - this.startTime >= this.timeLimit; }
  
  countMySquares(board) {
    let my = this.color === "R" ? -1 : -2, cnt = 0;
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++)
        if (board[i][j] === my) cnt++;
    return cnt;
  }
  
  countOpponentSquares(board) {
    let opp = this.color === "R" ? -2 : -1, cnt = 0;
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++)
        if (board[i][j] === opp) cnt++;
    return cnt;
  }
  
  randomMove(moves) { return moves[Math.floor(Math.random() * moves.length)]; }
  movesEqual(m1, m2) { return m1[0] === m2[0] && m1[1] === m2[1] && m1[2] === m2[2]; }

  boxesCompleted(board, r, c, s) {
    let gain = 0;
    if (this.count(board[r][c] | (1 << s)) === 4) gain++;
    const [dr, dc, os] = this.dirs[s];
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < this.size && nc >= 0 && nc < this.size && board[nr][nc] >= 0)
      if (this.count(board[nr][nc] | (1 << os)) === 4) gain++;
    return gain;
  }

  isSafe(board, r, c, s) {
    let v = board[r][c];
    if (v < 0) return false;
    let nv = v | (1 << s);
    if (this.count(nv) === 3) return false;
    let [dr, dc, os] = this.dirs[s];
    let nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < this.size && nc >= 0 && nc < this.size && board[nr][nc] >= 0) {
      let nnv = board[nr][nc] | (1 << os);
      if (this.count(nnv) === 3) return false;
    }
    return true;
  }

  getChainLengths(board) {
    let n = this.size, visited = Array.from({ length: n }, () => new Uint8Array(n));
    let lengths = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (visited[i][j]) continue;
        if (board[i][j] < 0) continue;
        if (this.count(board[i][j]) !== 2) continue;
        let queue = [[i, j]];
        visited[i][j] = 1;
        let size = 0;
        while (queue.length) {
          let [r, c] = queue.pop();
          size++;
          for (let s = 0; s < 4; s++) {
            if (board[r][c] & (1 << s)) continue;
            let [dr, dc] = this.dirs[s];
            let nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
            if (visited[nr][nc]) continue;
            if (board[nr][nc] < 0) continue;
            if (this.count(board[nr][nc]) !== 2) continue;
            visited[nr][nc] = 1;
            queue.push([nr, nc]);
          }
        }
        lengths.push(size);
      }
    }
    return lengths;
  }

  endgameTheoremScore(board) {
    const chains = this.getChainLengths(board);
    if (chains.length === 0) return null;
    chains.sort((a, b) => a - b);
    let total = 0, turn = 0;
    for (let len of chains) {
      if (turn === 0) total -= len;
      else total += len;
      turn = 1 - turn;
    }
    return total;
  }

  countSafeMoves(board) {
    let moves = this.boardUtil.valid_moves(board);
    let safeCount = 0;
    for (let m of moves)
      if (this.isSafe(board, m[0], m[1], m[2])) safeCount++;
    return safeCount;
  }

  getAntiSafeMove(board, moves) {
    let bestMove = moves[0];
    let minOppSafe = Infinity;
    const candidates = moves.slice(0, Math.min(20, moves.length));
    for (let move of candidates) {
      let nb = this.clone(board);
      let myColorInt = this.color === "R" ? -1 : -2;
      this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
      let oppSafe = this.countSafeMoves(nb);
      if (oppSafe < minOppSafe) {
        minOppSafe = oppSafe;
        bestMove = move;
      }
      if (minOppSafe === 0) break;
    }
    return bestMove;
  }

  endgameMoveLarge(board, moves) {
    let bestMove = moves[0];
    let bestDiff = -Infinity;
    for (let move of moves) {
      let nb = this.clone(board);
      let myColorInt = this.color === "R" ? -1 : -2;
      this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
      let diff = this.endgameTheoremScore(nb);
      if (diff === null) diff = this.countMySquares(nb) - this.countOpponentSquares(nb);
      if (diff > bestDiff) {
        bestDiff = diff;
        bestMove = move;
      }
    }
    return bestMove;
  }

  evaluateFast(board) {
    let my = this.color === "R" ? -1 : -2, opp = this.color === "R" ? -2 : -1;
    let mine = 0, theirs = 0;
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++) {
        let v = board[i][j];
        if (v === my) mine++;
        else if (v === opp) theirs++;
      }
    return (mine - theirs) * 100;
  }

  evaluate(board) {
    let myColor = this.color === "R" ? -1 : -2;
    let oppColor = this.color === "R" ? -2 : -1;
    let mine = 0, opp = 0;
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        let v = board[i][j];
        if (v === myColor) mine++;
        else if (v === oppColor) opp++;
      }
    }
    let score = (mine - opp) * 100;
    let remainingMoves = this.countRemainingMoves(board);
    if (remainingMoves < this.totalMoves * 0.7) {
      let structures = this.detectStructuresFast(board);
      let chainPenalty = structures.chains * 3 + structures.loops * 2;
      let longChains = 0;
      for (let c of structures.chainsList) if (c.length >= 3) longChains++;
      if (longChains % 2 === 1) chainPenalty -= 15;
      else chainPenalty += 15;
      score -= chainPenalty;
    }
    if (this.size > 8) {
      let corners = [[0,0], [0,this.size-1], [this.size-1,0], [this.size-1,this.size-1]];
      for (let [r,c] of corners) {
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            let nr = r+dr, nc = c+dc;
            if (nr>=0 && nr<this.size && nc>=0 && nc<this.size) {
              if (board[nr][nc] === myColor) score += 5;
              else if (board[nr][nc] === oppColor) score -= 5;
            }
          }
        }
      }
    }
    return score;
  }

  detectStructuresFast(board) {
    let n = this.size;
    let visited = Array.from({ length: n }, () => new Uint8Array(n));
    let chains = 0, loops = 0, chainsList = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (visited[i][j]) continue;
        if (board[i][j] < 0) continue;
        if (this.count(board[i][j]) !== 2) continue;
        let queue = [[i, j]];
        visited[i][j] = 1;
        let component = [];
        while (queue.length) {
          let [r, c] = queue.pop();
          component.push([r, c]);
          for (let s = 0; s < 4; s++) {
            if (board[r][c] & (1 << s)) continue;
            let [dr, dc] = this.dirs[s];
            let nr = r+dr, nc = c+dc;
            if (nr<0 || nr>=n || nc<0 || nc>=n) continue;
            if (visited[nr][nc]) continue;
            if (board[nr][nc] < 0) continue;
            if (this.count(board[nr][nc]) !== 2) continue;
            visited[nr][nc] = 1;
            queue.push([nr, nc]);
          }
        }
        let isLoop = true;
        for (let [r,c] of component) {
          let deg = 0;
          for (let s = 0; s < 4; s++) {
            if (board[r][c] & (1 << s)) continue;
            let [dr, dc] = this.dirs[s];
            let nr = r+dr, nc = c+dc;
            if (nr<0 || nr>=n || nc<0 || nc>=n) continue;
            if (board[nr][nc] < 0) continue;
            if (this.count(board[nr][nc]) === 2) deg++;
          }
          if (deg !== 2) { isLoop = false; break; }
        }
        if (isLoop) loops++;
        else { chains++; chainsList.push(component); }
      }
    }
    return { chains, loops, chainsList };
  }

  countRemainingMoves(board) {
    let remaining = 0;
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++)
        if (board[i][j] >= 0)
          remaining += this.remainingFromCell[board[i][j] & 15];
    return remaining;
  }

  movePriority(board, move) {
    let [r,c,s] = move;
    let gain = this.boxesCompleted(board,r,c,s);
    if (gain > 0) return 10000 + gain*1000;
    if (this.isSafe(board,r,c,s)) return 5000;
    let penalty = 0;
    let nv = board[r][c] | (1<<s);
    if (this.count(nv) === 3) penalty++;
    let [dr,dc,os] = this.dirs[s];
    let nr = r+dr, nc = c+dc;
    if (nr>=0 && nr<this.size && nc>=0 && nc<this.size && board[nr][nc]>=0) {
      let nnv = board[nr][nc] | (1<<os);
      if (this.count(nnv) === 3) penalty++;
    }
    return -penalty * 100;
  }

  orderMoves(board, moves, depth=0) {
    const killers = this.killerMoves[depth] || [];
    const prevMove = this.previousBestMove;
    return [...moves].sort((a,b) => {
      if (prevMove && this.movesEqual(a,prevMove)) return -1;
      if (prevMove && this.movesEqual(b,prevMove)) return 1;
      const aIsKiller = killers.some(k => this.movesEqual(a,k));
      const bIsKiller = killers.some(k => this.movesEqual(b,k));
      if (aIsKiller !== bIsKiller) return aIsKiller ? -1 : 1;
      return this.movePriority(board,b) - this.movePriority(board,a);
    });
  }

  updateKillerMoves(move, depth) {
    const killers = this.killerMoves[depth];
    if (!killers.some(k => this.movesEqual(k,move))) {
      killers.unshift(move);
      if (killers.length > 2) killers.pop();
    }
  }

  getBoardHash(board) {
    let hash = 0;
    const prime = 0x9e3779b9;
    for (let i = 0; i < this.size; i++)
      for (let j = 0; j < this.size; j++) {
        let val = board[i][j];
        if (val < 0) val = -val + 10;
        hash = ((hash << 5) - hash) ^ (val + i*this.size + j);
        hash ^= (hash >>> 16);
      }
    return hash;
  }

  quiescence(board, alpha, beta, isMaximizing, startTime, timeLimit) {
    if (Date.now() - startTime > timeLimit) return this.evaluateFast(board);
    let standPat = this.evaluate(board);
    if (isMaximizing) {
      if (standPat >= beta) return beta;
      if (standPat > alpha) alpha = standPat;
    } else {
      if (standPat <= alpha) return alpha;
      if (standPat < beta) beta = standPat;
    }
    let captureMoves = [];
    let allMoves = this.boardUtil.valid_moves(board);
    for (let m of allMoves)
      if (this.boxesCompleted(board, m[0], m[1], m[2]) > 0)
        captureMoves.push(m);
    if (captureMoves.length === 0) return standPat;
    captureMoves.sort((a,b) => {
      let ga = this.boxesCompleted(board, a[0], a[1], a[2]);
      let gb = this.boxesCompleted(board, b[0], b[1], b[2]);
      return gb - ga;
    });
    let myColorInt = this.color === "R" ? -1 : -2;
    let oppColorInt = this.color === "R" ? -2 : -1;
    for (let move of captureMoves) {
      let gain = this.boxesCompleted(board, move[0], move[1], move[2]);
      let nb = this.clone(board);
      let currentPlayer = isMaximizing ? myColorInt : oppColorInt;
      this.boardUtil.move(nb, move[0], move[1], move[2], currentPlayer);
      let score = this.quiescence(nb, alpha, beta, isMaximizing, startTime, timeLimit);
      if (isMaximizing) {
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
      } else {
        if (score <= alpha) return alpha;
        if (score < beta) beta = score;
      }
    }
    return isMaximizing ? alpha : beta;
  }

  alphabeta(board, depth, alpha, beta, isMaximizing, startTime, timeLimit, lastCapture) {
    if (Date.now() - startTime > timeLimit) return this.evaluateFast(board);
    if (depth === 0) return this.quiescence(board, alpha, beta, isMaximizing, startTime, timeLimit);
    let moves = this.boardUtil.valid_moves(board);
    if (moves.length === 0) return this.evaluateFast(board);
    moves = this.orderMoves(board, moves, depth);
    let limit = depth <= 2 ? Math.min(18, moves.length) : Math.min(12, moves.length);
    moves = moves.slice(0, limit);
    const hash = this.getBoardHash(board);
    const entry = this.transTable.get(hash);
    if (entry && entry.depth >= depth) {
      if (entry.type === 'exact') return entry.value;
      if (entry.type === 'lower' && entry.value > alpha) alpha = entry.value;
      if (entry.type === 'upper' && entry.value < beta) beta = entry.value;
      if (alpha >= beta) return entry.value;
    }
    let myColorInt = this.color === "R" ? -1 : -2;
    let oppColorInt = this.color === "R" ? -2 : -1;
    let value, type, bestMove = null;
    if (isMaximizing) {
      value = -Infinity;
      for (let move of moves) {
        let gain = this.boxesCompleted(board, move[0], move[1], move[2]);
        let nb = this.clone(board);
        this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
        let nextMax = gain > 0;
        let score = this.alphabeta(nb, depth-1, alpha, beta, nextMax, startTime, timeLimit, gain>0);
        if (score > value) { value = score; bestMove = move; }
        if (score > alpha) { alpha = score; this.updateKillerMoves(move, depth); }
        if (beta <= alpha) break;
      }
      type = (value <= alpha) ? 'upper' : (value >= beta) ? 'lower' : 'exact';
    } else {
      value = Infinity;
      for (let move of moves) {
        let gain = this.boxesCompleted(board, move[0], move[1], move[2]);
        let nb = this.clone(board);
        this.boardUtil.move(nb, move[0], move[1], move[2], oppColorInt);
        let nextMax = gain > 0;
        let score = this.alphabeta(nb, depth-1, alpha, beta, nextMax, startTime, timeLimit, gain>0);
        if (score < value) { value = score; bestMove = move; }
        if (score < beta) { beta = score; this.updateKillerMoves(move, depth); }
        if (beta <= alpha) break;
      }
      type = (value <= alpha) ? 'upper' : (value >= beta) ? 'lower' : 'exact';
    }
    this.transTable.set(hash, { depth, value, type, bestMove });
    return value;
  }

  selectRobustMove(board, candidateMoves, scores) {
    if (!candidateMoves || candidateMoves.length === 0) return null;
    let maxScore = Math.max(...scores);
    let bestIndices = [];
    for (let i=0; i<scores.length; i++) if (scores[i] === maxScore) bestIndices.push(i);
    if (bestIndices.length === 1) return candidateMoves[bestIndices[0]];
    let bestMove = candidateMoves[bestIndices[0]];
    let maxVariance = -Infinity;
    for (let idx of bestIndices) {
      let move = candidateMoves[idx];
      let nb = this.clone(board);
      let myColorInt = this.color === "R" ? -1 : -2;
      this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
      let variance = this.boardUtil.valid_moves(nb).length;
      if (variance > maxVariance) { maxVariance = variance; bestMove = move; }
    }
    return bestMove;
  }

  endgameSearch(board, moves) {
    let bestMove = moves[0];
    let bestScore = -Infinity;
    let start = this.startTime;
    let budget = Math.min(this.timeLimit * 0.95, this.size <= 10 ? 600 : (this.size <= 20 ? 500 : 400));
    let depth = this.getEndgameDepth(moves.length);
    for (let move of moves) {
      if (Date.now() - start > budget) break;
      let [r,c,s] = move;
      let gain = this.boxesCompleted(board, r, c, s);
      let nb = this.clone(board);
      let myColorInt = this.color === "R" ? -1 : -2;
      this.boardUtil.move(nb, r, c, s, myColorInt);
      let score = this.alphabeta(nb, depth, -Infinity, Infinity, gain>0, start, budget, gain>0);
      if (score > bestScore) { bestScore = score; bestMove = move; }
    }
    return bestMove;
  }

  getEndgameDepth(movesLeft) {
    if (this.size <= 10) {
      if (movesLeft < 12) return 16;
      if (movesLeft < 20) return 14;
      return 12;
    } else if (this.size <= 15) {
      if (movesLeft < 15) return 14;
      if (movesLeft < 25) return 12;
      return 10;
    } else if (this.size <= 20) {
      if (movesLeft < 20) return 13;
      if (movesLeft < 35) return 11;
      return 9;
    } else {
      if (movesLeft < 25) return 18;
      if (movesLeft < 40) return 16;
      if (movesLeft < 60) return 14;
      return 12;
    }
  }

  getNormalDepth(timeRemaining, movesLeft) {
    let baseDepth;
    if (this.size <= 10) baseDepth = 10;
    else if (this.size <= 15) baseDepth = 9;
    else if (this.size <= 20) baseDepth = 8;
    else baseDepth = 7;
    if (timeRemaining < 3000) baseDepth = Math.max(4, baseDepth-1);
    if (timeRemaining < 1500) baseDepth = Math.max(3, baseDepth-1);
    if (movesLeft < 40) baseDepth += 2;
    if (movesLeft < 20) baseDepth += 2;
    return baseDepth;
  }

  getMaxPossibleDepth(movesLeft, timeRemaining) {
    let maxDepth;
    if (this.size <= 10) maxDepth = 15;
    else if (this.size <= 15) maxDepth = 14;
    else if (this.size <= 20) maxDepth = 13;
    else maxDepth = 20;

    if (this.size > 20) {
      if (timeRemaining < 8000) maxDepth = Math.min(maxDepth, 18);
      if (timeRemaining < 4000) maxDepth = Math.min(maxDepth, 15);
      if (timeRemaining < 2000) maxDepth = Math.min(maxDepth, 12);
      if (timeRemaining < 1000) maxDepth = Math.min(maxDepth, 10);
    } else {
      if (timeRemaining < 4000) maxDepth = Math.min(maxDepth, 10);
      if (timeRemaining < 2000) maxDepth = Math.min(maxDepth, 8);
      if (timeRemaining < 1000) maxDepth = Math.min(maxDepth, 6);
    }

    if (movesLeft < 40) maxDepth = Math.min(maxDepth + (this.size > 20 ? 4 : 2), 24);
    if (movesLeft < 20) maxDepth = Math.min(maxDepth + (this.size > 20 ? 4 : 2), 26);
    return Math.max(4, maxDepth);
  }

  computeTimeLimitSmall(timeRemaining, movesLeft) {
    let ratio;
    let maxLimit;
    if (this.size <= 20) {
      ratio = 0.12;
      if (movesLeft < this.totalMoves * 0.3) ratio = 0.25;
      else if (movesLeft < this.totalMoves * 0.6) ratio = 0.18;
      maxLimit = this.size <= 10 ? 600 : 900;
    } else {
      ratio = 0.20;
      if (movesLeft < this.totalMoves * 0.3) ratio = 0.35;
      else if (movesLeft < this.totalMoves * 0.6) ratio = 0.28;
      maxLimit = 2500;
    }
    let limit = Math.min(timeRemaining * ratio, maxLimit);
    return Math.max(this.size <= 20 ? 60 : 100, limit);
  }

  
  compute(board, timeRemaining) {
    this.size = board.length;
    this.totalMoves = 2 * this.size * (this.size + 1);
    this.initialTime = timeRemaining;

    let moves = this.boardUtil.valid_moves(board);
    if (moves.length === 0) return [0, 0, 0];

    this.startTime = Date.now();
    this.timeLimit = this.computeTimeLimitSmall(timeRemaining, moves.length);
    this.transTable.clear();
    this.killerMoves = new Array(64).fill().map(() => []);
    this.previousBestMove = null;

    let totalSquares = this.size * this.size;
    let winCondition = Math.floor(totalSquares / 2) + 1;
    if (this.countMySquares(board) >= winCondition) return this.randomMove(moves);
    if (this.countOpponentSquares(board) >= winCondition) return this.randomMove(moves);

    if (moves.length > this.totalMoves * 0.35) {
      for (let move of moves) {
        let nb = this.clone(board);
        let myColorInt = this.color === "R" ? -1 : -2;
        this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
        if (this.countMySquares(nb) >= winCondition) return move;
      }
    }

    let captures = [];
    for (let m of moves) {
      if (this.boxesCompleted(board, m[0], m[1], m[2]) > 0) captures.push(m);
    }
    if (captures.length > 0) {
      if (captures.length > 1 && moves.length < 100) return this.endgameSearch(board, captures);
      let best = captures[0];
      let bestGain = 0;
      for (let m of captures) {
        let g = this.boxesCompleted(board, m[0], m[1], m[2]);
        if (g > bestGain) {
          bestGain = g;
          best = m;
        }
      }
      return best;
    }

    let safe = [];
    for (let m of moves) {
      if (this.isSafe(board, m[0], m[1], m[2])) safe.push(m);
    }
    if (safe.length > 0) {
      if (safe.length < 40 || moves.length < 70) return this.endgameSearch(board, safe);
      let ordered = this.orderMoves(board, safe);
      return ordered[0];
    }

    let baseOrderedMoves = this.orderMoves(board, moves);
    let bestMove = baseOrderedMoves[0];
    let maxDepth = this.getMaxPossibleDepth(moves.length, timeRemaining);
    let scoresAtDepth = [];

    for (let depth = 1; depth <= maxDepth; depth++) {
      if (this.outOfTime()) break;

      let currentBestMove = bestMove;
      let bestScore = -Infinity;
      let rootLimit = Math.min(baseOrderedMoves.length,
          this.size <= 10 ? 22 :
          (this.size <= 20 ? 20 :
          (this.size <= 30 ? 24 : 28)));

      let currentOrdered = [...baseOrderedMoves];
      if (this.previousBestMove) {
        currentOrdered = [this.previousBestMove, ...currentOrdered.filter(m => !this.movesEqual(m, this.previousBestMove))];
      }

      let moveScores = [];
      for (let i = 0; i < rootLimit; i++) {
        if (this.outOfTime()) break;
        let move = currentOrdered[i];
        let gain = this.boxesCompleted(board, move[0], move[1], move[2]);
        let nb = this.clone(board);
        let myColorInt = this.color === "R" ? -1 : -2;
        this.boardUtil.move(nb, move[0], move[1], move[2], myColorInt);
        let score = this.alphabeta(nb, depth - 1, -Infinity, Infinity, gain > 0, this.startTime, this.timeLimit, gain > 0);
        moveScores.push(score);
        if (score > bestScore) {
          bestScore = score;
          currentBestMove = move;
        }
      }
      scoresAtDepth.push({ moves: currentOrdered.slice(0, rootLimit), scores: moveScores });
      bestMove = currentBestMove;
      this.previousBestMove = bestMove;
    }

    if (scoresAtDepth.length > 0) {
      let last = scoresAtDepth[scoresAtDepth.length - 1];
      if (last.moves.length === last.scores.length) {
        bestMove = this.selectRobustMove(board, last.moves, last.scores);
      }
    }

    if (safe.length === 0 && captures.length === 0) {
      return this.endgameSearch(board, moves);
    }
    return bestMove;
  }
}