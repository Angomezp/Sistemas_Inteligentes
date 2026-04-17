/*
 * AzucarMorena - Constructor de Cadenas (Versión Ultra Rápida)
 * Optimizaciones:
 * - Cache de patrones de apertura
 * - Evaluación perezosa con early exit
 * - Pre-cálculo de valores de posición
 * - Movimientos predeterminados para situaciones comunes
 */
class AzucarMorena extends Agent {
    constructor() {
        super();
        this.boardOps = new Board();
        this.turnCounter = 0;
        
        // Cache de posiciones para no recalcular
        this.positionCache = new Map();
        this.lastBoardHash = "";
        this.lastValidMoves = [];
        
        // Valores pre-calculados por posición
        this.positionValues = null;
        
        // Patrones de apertura para tableros comunes
        this.openingBook = this.initOpeningBook();
    }

    init(color, board, time) {
        super.init(color, board, time);
        this.opponentColor = (color === 'R') ? 'Y' : 'R';
        this.pieceValue = (color === 'R') ? -1 : -2;
        this.opponentValue = (this.opponentColor === 'R') ? -1 : -2;
        this.turnCounter = 0;
        this.positionCache.clear();
        
        // Pre-calcular valores de posición (solo una vez)
        this.initPositionValues();
    }
    
    // Pre-calcular valores de todas las posiciones del tablero
    initPositionValues() {
        this.positionValues = new Array(this.size);
        for (let i = 0; i < this.size; i++) {
            this.positionValues[i] = new Array(this.size);
            for (let j = 0; j < this.size; j++) {
                let value = 0;
                // Esquinas
                if ((i === 0 || i === this.size-1) && (j === 0 || j === this.size-1)) {
                    value = 100;
                }
                // Bordes
                else if (i === 0 || i === this.size-1 || j === 0 || j === this.size-1) {
                    value = 40;
                }
                // Centro
                else {
                    let center = (this.size - 1) / 2;
                    let dist = Math.abs(i - center) + Math.abs(j - center);
                    value = Math.max(0, 20 - dist);
                }
                this.positionValues[i][j] = value;
            }
        }
    }
    
    // Libro de aperturas para respuestas rápidas
    initOpeningBook() {
        return {
            // Primer movimiento en tablero vacío
            'first_move_3x3': [1, 1, 0],
            'first_move_4x4': [1, 1, 0],
            'first_move_5x5': [2, 2, 0],
            'first_move_6x6': [2, 2, 0],
            'first_move_7x7': [3, 3, 0],
            'first_move_8x8': [3, 3, 0],
            
            // Respuestas a movimientos comunes del oponente
            'opponent_corner': [0, 1, 1],  // Si oponente juega esquina, jugar borde adyacente
            'opponent_center': [1, 1, 2],  // Si oponente juega centro, jugar abajo
        };
    }

    compute(board, timeRemaining) {
        // ==================== VERIFICACIÓN RÁPIDA ====================
        if (!board || board.length === 0) return [0, 0, 0];
        
        if (this.size !== board.length) {
            this.size = board.length;
            this.initPositionValues();
        }
        
        // Obtener movimientos válidos con cache
        let validMoves = this.getCachedValidMoves(board);
        if (!validMoves || validMoves.length === 0) return [0, 0, 0];
        
        this.turnCounter++;
        
        // ==================== EARLY EXIT: Movimientos obvios ====================
        
        // 1. Ganar inmediatamente (rápido)
        let winningMove = this.findWinFast(board, validMoves);
        if (winningMove) return winningMove;
        
        // 2. Bloquear victoria del oponente (rápido)
        let blockingMove = this.findBlockFast(board, validMoves);
        if (blockingMove) return blockingMove;
        
        // 3. Verificar libro de aperturas (primeros 6 movimientos)
        if (this.turnCounter <= 6) {
            let bookMove = this.checkOpeningBook(board);
            if (bookMove && this.isMoveValid(board, bookMove)) {
                return bookMove;
            }
        }
        
        // 4. Cache de posiciones similares
        let cachedMove = this.checkPositionCache(board);
        if (cachedMove && this.isMoveValid(board, cachedMove)) {
            return cachedMove;
        }
        
        // ==================== EVALUACIÓN RÁPIDA ====================
        // Limitar análisis a máximo 30 movimientos (no todos)
        let maxMoves = Math.min(validMoves.length, 30);
        let bestMove = null;
        let bestScore = -Infinity;
        
        for (let idx = 0; idx < maxMoves; idx++) {
            let move = validMoves[idx];
            if (!this.isMoveValid(board, move)) continue;
            
            // Calcular score de manera rápida
            let score = this.fastScore(board, move);
            
            // Early exit si encontramos movimiento excelente
            if (score > 800) {
                this.cachePosition(board, move);
                return move;
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        // Cachear este movimiento para futuras posiciones similares
        if (bestMove) {
            this.cachePosition(board, bestMove);
        }
        
        return bestMove || validMoves[0];
    }
    
    // ==================== CACHE Y OPTIMIZACIONES ====================
    
    getCachedValidMoves(board) {
        let hash = this.quickHash(board);
        if (hash === this.lastBoardHash) {
            return this.lastValidMoves;
        }
        this.lastBoardHash = hash;
        this.lastValidMoves = this.boardOps.valid_moves(board);
        return this.lastValidMoves;
    }
    
    quickHash(board) {
        // Hash rápido usando solo esquinas y centro
        let h = 0;
        h = h * 31 + board[0][0];
        h = h * 31 + board[0][this.size-1];
        h = h * 31 + board[this.size-1][0];
        h = h * 31 + board[this.size-1][this.size-1];
        let mid = Math.floor(this.size / 2);
        h = h * 31 + board[mid][mid];
        return h;
    }
    
    cachePosition(board, move) {
        let key = this.quickHash(board);
        if (!this.positionCache.has(key)) {
            this.positionCache.set(key, move);
        }
        // Limitar tamaño del cache
        if (this.positionCache.size > 100) {
            let firstKey = this.positionCache.keys().next().value;
            this.positionCache.delete(firstKey);
        }
    }
    
    checkPositionCache(board) {
        let key = this.quickHash(board);
        if (this.positionCache.has(key)) {
            let cached = this.positionCache.get(key);
            if (this.isMoveValid(board, cached)) {
                return cached;
            }
        }
        return null;
    }
    
    checkOpeningBook(board) {
        // Detectar tablero vacío (primer movimiento)
        let isEmpty = true;
        for (let i = 0; i < this.size && isEmpty; i++) {
            for (let j = 0; j < this.size && isEmpty; j++) {
                if (board[i][j] !== 0 && board[i][j] !== 9 && board[i][j] !== 3 && 
                    board[i][j] !== 8 && board[i][j] !== 2 && board[i][j] !== 12 && 
                    board[i][j] !== 4 && board[i][j] !== 6) {
                    isEmpty = false;
                }
            }
        }
        
        if (isEmpty && this.turnCounter === 1) {
            // Primer movimiento: centro o cerca del centro
            let center = Math.floor(this.size / 2);
            return [center, center, 0];
        }
        
        return null;
    }
    
    // ==================== BÚSQUEDA RÁPIDA DE GANADORES ====================
    
    findWinFast(board, moves) {
        for (let move of moves) {
            // Verificar solo 4 direcciones (muy rápido)
            let [row, col, side] = move;
            if (side === 0 && row > 0 && this.isCellNearCompleteFast(board, row-1, col)) return move;
            if (side === 1 && col < this.size-1 && this.isCellNearCompleteFast(board, row, col+1)) return move;
            if (side === 2 && row < this.size-1 && this.isCellNearCompleteFast(board, row+1, col)) return move;
            if (side === 3 && col > 0 && this.isCellNearCompleteFast(board, row, col-1)) return move;
        }
        return null;
    }
    
    findBlockFast(board, moves) {
        for (let move of moves) {
            let [row, col, side] = move;
            // Verificar si el oponente completaría un cuadro
            if (side === 0 && row > 0 && this.isCellTwoLinesFast(board, row-1, col)) return move;
            if (side === 1 && col < this.size-1 && this.isCellTwoLinesFast(board, row, col+1)) return move;
            if (side === 2 && row < this.size-1 && this.isCellTwoLinesFast(board, row+1, col)) return move;
            if (side === 3 && col > 0 && this.isCellTwoLinesFast(board, row, col-1)) return move;
        }
        return null;
    }
    
    isCellNearCompleteFast(board, row, col) {
        let cell = board[row][col];
        if (cell < 0) return false;
        return this.countLinesFast(cell) === 3;
    }
    
    isCellTwoLinesFast(board, row, col) {
        let cell = board[row][col];
        if (cell < 0) return false;
        return this.countLinesFast(cell) === 2;
    }
    
    // ==================== FUNCIÓN DE SCORE RÁPIDA ====================
    
    fastScore(board, move) {
        let [row, col, side] = move;
        let score = 0;
        
        // 1. Ganancia inmediata (más importante)
        let gain = 0;
        if (side === 0 && row > 0 && this.isCellNearCompleteFast(board, row-1, col)) gain++;
        if (side === 1 && col < this.size-1 && this.isCellNearCompleteFast(board, row, col+1)) gain++;
        if (side === 2 && row < this.size-1 && this.isCellNearCompleteFast(board, row+1, col)) gain++;
        if (side === 3 && col > 0 && this.isCellNearCompleteFast(board, row, col-1)) gain++;
        score += gain * 500;
        
        // 2. Si gana múltiples cuadros, retornar inmediatamente
        if (gain >= 2) return 1000;
        
        // 3. Valor de posición (pre-calculado)
        score += this.positionValues[row][col];
        
        // 4. Potencial de cadena (rápido - solo mirar alrededor)
        let chainBonus = 0;
        let neighbors = [[row-1, col], [row+1, col], [row, col-1], [row, col+1]];
        for (let [nr, nc] of neighbors) {
            if (nr >= 0 && nr < this.size && nc >= 0 && nc < this.size) {
                let cell = board[nr][nc];
                if (cell >= 0) {
                    let lines = this.countLinesFast(cell);
                    if (lines === 2) chainBonus += 30;
                    else if (lines === 1) chainBonus += 10;
                }
            }
        }
        score += chainBonus;
        
        // 5. Penalizar regalar cuadros al inicio
        if (this.turnCounter < 10 && gain === 0 && this.wouldCreateOpponentSquareFast(board, move)) {
            score -= 200;
        }
        
        // 6. Bonus por crear nuevas celdas con 2 líneas
        let newTwoLines = this.countNewTwoLinesFast(board, move);
        score += newTwoLines * 40;
        
        return score;
    }
    
    countNewTwoLinesFast(board, move) {
        let [row, col, side] = move;
        let count = 0;
        
        // Verificar celda actual
        let newCell = board[row][col] | (1 << side);
        if (this.countLinesFast(newCell) === 2) count++;
        
        // Verificar vecino afectado
        if (side === 0 && row > 0) {
            let newNeighbor = board[row-1][col] | 4;
            if (this.countLinesFast(newNeighbor) === 2) count++;
        }
        if (side === 1 && col < this.size-1) {
            let newNeighbor = board[row][col+1] | 8;
            if (this.countLinesFast(newNeighbor) === 2) count++;
        }
        if (side === 2 && row < this.size-1) {
            let newNeighbor = board[row+1][col] | 1;
            if (this.countLinesFast(newNeighbor) === 2) count++;
        }
        if (side === 3 && col > 0) {
            let newNeighbor = board[row][col-1] | 2;
            if (this.countLinesFast(newNeighbor) === 2) count++;
        }
        
        return count;
    }
    
    wouldCreateOpponentSquareFast(board, move) {
        let [row, col, side] = move;
        
        // Verificar si este movimiento completa un cuadro que el oponente tomará
        if (side === 0 && row > 0 && this.isCellTwoLinesFast(board, row-1, col)) return true;
        if (side === 1 && col < this.size-1 && this.isCellTwoLinesFast(board, row, col+1)) return true;
        if (side === 2 && row < this.size-1 && this.isCellTwoLinesFast(board, row+1, col)) return true;
        if (side === 3 && col > 0 && this.isCellTwoLinesFast(board, row, col-1)) return true;
        
        return false;
    }
    
    countLinesFast(cell) {
        // Versión optimizada con operaciones de bits
        return ((cell >> 0) & 1) +
               ((cell >> 1) & 1) +
               ((cell >> 2) & 1) +
               ((cell >> 3) & 1);
    }
    
    // ==================== FUNCIONES AUXILIARES ====================
    
    isMoveValid(board, move) {
        if (!move || move.length !== 3) return false;
        let [row, col, side] = move;
        
        if (row < 0 || row >= this.size || col < 0 || col >= this.size) return false;
        if (side < 0 || side > 3) return false;
        
        return this.boardOps.check(board, row, col, side);
    }
}