/*
 * AzucarMorena PRO+ LIGHT v2.3
 * Ajuste: minimax activo en último 40% para tableros ≤20, y último 30% para >20.
 */

class AzucarMorena extends Agent {

    constructor() {
        super();
        this.boardOps = new Board();
        this.size = 0;
        this.color = "R";
        this.opponentColor = "Y";
        this.totalMoves = 0;
        this.movesDone = 0;
        this.randomPhaseLimit = 0;
        this.minimaxPhaseStart = 0;
    }

    init(color, board, time) {
        super.init(color, board, time);
        this.color = color;
        this.opponentColor = (color === "R") ? "Y" : "R";
        this.size = board.length;
        this.totalMoves = 2 * this.size * (this.size + 1);
        this.randomPhaseLimit = Math.floor(this.totalMoves * 0.04);

        // Ajuste del umbral de minimax según tamaño del tablero
        if (this.size <= 20) {
            // Tableros pequeños/medianos: minimax en el último 40% de las jugadas
            this.minimaxPhaseStart = Math.floor(this.totalMoves * 0.60);
        } else {
            // Tableros grandes: minimax en el último 30%
            this.minimaxPhaseStart = Math.floor(this.totalMoves * 0.70);
        }
    }

    compute(board, timeRemaining) {
        let moves = this.boardOps.valid_moves(board);
        if (!moves || moves.length === 0) return [0,0,0];

        this.movesDone = this.totalMoves - moves.length;

        // Apertura aleatoria
        if (this.movesDone < this.randomPhaseLimit) {
            return moves[Math.floor(Math.random() * moves.length)];
        }

        // Final: minimax según umbral configurado
        if (this.movesDone >= this.minimaxPhaseStart) {
            return this.minimaxMove(board, moves);
        }

        // Gana inmediato si es posible
        let win = this.findImmediateWin(board, moves);
        if (win) return win;

        // Medio juego mejorado
        return this.findBestMidMove(board, moves);
    }

    // =====================================================
    // UTILIDADES BÁSICAS (sin cambios)
    // =====================================================
    countLines(cell) {
        return ((cell & 1) ? 1 : 0) +
               ((cell & 2) ? 1 : 0) +
               ((cell & 4) ? 1 : 0) +
               ((cell & 8) ? 1 : 0);
    }

    countPlayerSquares(board, color) {
        let target = color === "R" ? -1 : -2;
        let total = 0;
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                if (board[i][j] === target) total++;
            }
        }
        return total;
    }

    applyMove(board, move, color) {
        let [r, c, s] = move;
        let mark = color === "R" ? -1 : -2;
        this.boardOps.move(board, r, c, s, mark);
    }

    applyMoveAndGetSquares(board, move, color) {
        let before = this.countPlayerSquares(board, color);
        this.applyMove(board, move, color);
        let after = this.countPlayerSquares(board, color);
        return after - before;
    }

    // =====================================================
    // GANAR INMEDIATAMENTE
    // =====================================================
    findImmediateWin(board, moves) {
        let best = null;
        let bestGain = 0;
        for (let move of moves) {
            let gain = this.countSquaresCompleted(board, move);
            if (gain > bestGain) {
                bestGain = gain;
                best = move;
            }
        }
        return best;
    }

    countSquaresCompleted(board, move) {
        let temp = this.boardOps.clone(board);
        let before = this.countPlayerSquares(temp, this.color);
        this.applyMove(temp, move, this.color);
        let after = this.countPlayerSquares(temp, this.color);
        return after - before;
    }

    // =====================================================
    // MEDIO JUEGO MEJORADO
    // =====================================================
    findBestMidMove(board, moves) {
        let bestMove = moves[0];
        let bestScore = -Infinity;
        for (let move of moves) {
            let score = this.evaluateMidMove(board, move);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        return bestMove;
    }

    evaluateMidMove(board, move) {
        let [r, c, s] = move;
        let score = 0;

        // Ganancia inmediata de cajas
        let boxes = this.countSquaresCompleted(board, move);
        score += boxes * 1400;

        // Penalización por crear celdas de 3 líneas (regalar cajas)
        if (this.createsThirdLine(board, move)) score -= 400;

        // Penalización por crear múltiples amenazas (celdas de 3 líneas tras el movimiento)
        score -= this.multiThreat(board, move) * 150;

        // Bonus por crear una celda de 2 líneas (construir cadena)
        if (this.createsSecondLine(board, move)) score += 70;

        // Potencial de cadena (conectividad con vecinos)
        score += this.chainPotential(board, r, c) * 45;

        // Bonus si el movimiento permite una segunda captura inmediata (doble caja)
        if (boxes === 2) score += 300;
        if (boxes === 1 && this.followUpCapture(board, move)) score += 120;

        // Preferencia por bordes y esquinas
        if (r === 0 || c === 0 || r === this.size - 1 || c === this.size - 1) score += 8;
        if ((r === 0 || r === this.size - 1) && (c === 0 || c === this.size - 1)) score += 12;

        return score;
    }

    // Detecta si tras completar una caja, se puede capturar otra inmediatamente
    followUpCapture(board, move) {
        let temp = this.boardOps.clone(board);
        this.applyMove(temp, move, this.color);
        let nextMoves = this.boardOps.valid_moves(temp);
        for (let nm of nextMoves) {
            if (this.countSquaresCompleted(temp, nm) > 0) {
                return true;
            }
        }
        return false;
    }

    createsSecondLine(board, move) {
        let [r, c, s] = move;
        let cell = board[r][c];
        let next = cell | (1 << s);
        return (cell >= 0 && this.countLines(cell) === 1 && this.countLines(next) === 2);
    }

    createsThirdLine(board, move) {
        let [r, c, s] = move;
        let cell = board[r][c];
        let next = cell | (1 << s);
        return (cell >= 0 && this.countLines(cell) === 2 && this.countLines(next) === 3);
    }

    multiThreat(board, move) {
        let temp = this.boardOps.clone(board);
        this.applyMove(temp, move, this.color);
        let danger = 0;
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                let cell = temp[i][j];
                if (cell >= 0 && this.countLines(cell) === 3) danger++;
            }
        }
        return danger;
    }

    chainPotential(board, row, col) {
        let total = 0;
        let dirs = [[1,0],[-1,0],[0,1],[0,-1]];
        for (let d of dirs) {
            let nr = row + d[0];
            let nc = col + d[1];
            if (nr >= 0 && nc >= 0 && nr < this.size && nc < this.size) {
                let cell = board[nr][nc];
                if (cell >= 0) {
                    let lines = this.countLines(cell);
                    if (lines === 2) total += 3;
                    else if (lines === 1) total += 1;
                }
            }
        }
        return total;
    }

    // =====================================================
    // MINIMAX DINÁMICO Y ROBUSTO
    // =====================================================
    minimaxMove(board, moves) {
        let ordered = this.orderMoves(board, moves);
        let bestMove = ordered[0];
        let bestVal = -Infinity;

        // Profundidad según fase del juego
        let remaining = this.totalMoves - this.movesDone;
        let depth = 3;
        if (remaining <= 20) depth = 4;
        if (remaining <= 10) depth = 5;
        if (remaining <= 6) depth = 6;

        // Ancho de búsqueda adaptativo
        let width = Math.min(ordered.length, (depth <= 3 ? 6 : 4));

        for (let i = 0; i < width; i++) {
            let move = ordered[i];
            let temp = this.boardOps.clone(board);
            let gain = this.applyMoveAndGetSquares(temp, move, this.color);

            let val;
            if (gain > 0) {
                val = gain * 100 + this.minimaxLite(temp, depth - 1, -Infinity, Infinity, true);
            } else {
                val = this.minimaxLite(temp, depth - 1, -Infinity, Infinity, false);
            }

            if (val > bestVal) {
                bestVal = val;
                bestMove = move;
            }
        }
        return bestMove;
    }

    minimaxLite(board, depth, alpha, beta, maximizing) {
        let moves = this.boardOps.valid_moves(board);
        if (depth === 0 || moves.length === 0) {
            return this.fastEvaluate(board);
        }

        moves = this.orderMoves(board, moves);
        let width = Math.min(3, moves.length);

        let color = maximizing ? this.color : this.opponentColor;

        if (maximizing) {
            let best = -Infinity;
            for (let i = 0; i < width; i++) {
                let temp = this.boardOps.clone(board);
                let gain = this.applyMoveAndGetSquares(temp, moves[i], color);
                let val;
                if (gain > 0) {
                    val = gain * 70 + this.minimaxLite(temp, depth - 1, alpha, beta, true);
                } else {
                    val = this.minimaxLite(temp, depth - 1, alpha, beta, false);
                }
                best = Math.max(best, val);
                alpha = Math.max(alpha, best);
                if (beta <= alpha) break;
            }
            return best;
        } else {
            let best = Infinity;
            for (let i = 0; i < width; i++) {
                let temp = this.boardOps.clone(board);
                let gain = this.applyMoveAndGetSquares(temp, moves[i], color);
                let val;
                if (gain > 0) {
                    val = -gain * 70 + this.minimaxLite(temp, depth - 1, alpha, beta, false);
                } else {
                    val = this.minimaxLite(temp, depth - 1, alpha, beta, true);
                }
                best = Math.min(best, val);
                beta = Math.min(beta, best);
                if (beta <= alpha) break;
            }
            return best;
        }
    }

    // =====================================================
    // ORDENAMIENTO INTELIGENTE
    // =====================================================
    orderMoves(board, moves) {
        let scored = [];
        for (let move of moves) {
            let score = this.countSquaresCompleted(board, move) * 1000;
            if (this.createsThirdLine(board, move)) score -= 500;
            score += this.chainPotential(board, move[0], move[1]) * 30;
            if (this.createsSecondLine(board, move)) score += 80;
            score -= this.multiThreat(board, move) * 80;
            scored.push({ move: move, score: score });
        }
        scored.sort((a,b) => b.score - a.score);
        return scored.map(x => x.move);
    }

    // =====================================================
    // EVALUACIÓN MEJORADA (HEURÍSTICA)
    // =====================================================
    fastEvaluate(board) {
        let my = this.countPlayerSquares(board, this.color);
        let opp = this.countPlayerSquares(board, this.opponentColor);
        let value = (my - opp) * 150;

        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                let cell = board[i][j];
                if (cell >= 0) {
                    let lines = this.countLines(cell);
                    if (lines === 3) value -= 15;
                    else if (lines === 2) value += 3;
                }
                if (i === 0 || j === 0 || i === this.size-1 || j === this.size-1) value += 1;
                if ((i === 0 || i === this.size-1) && (j === 0 || j === this.size-1)) value += 2;
            }
        }
        return value;
    }
}