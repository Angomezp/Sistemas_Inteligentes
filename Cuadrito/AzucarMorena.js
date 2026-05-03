class AzucarMorena extends Agent {

    constructor() {

        super();

        this.boardOps = new Board();

        this.size = 0;

        this.color = "R";
        this.opponentColor = "Y";

        this.totalMoves = 0;
        this.movesDone = 0;

        this.initialTime = 0;

        this.startTime = 0;
        this.timeLimit = 0;

        this.FAST_MODE = false;
        this.ULTRA_FAST = false;
        this.BLITZ_20 = false;

        this.openingPercent = 0.05;
        this.minimaxStartPercent = 0.75;

        this.baseDepth = 3;
        this.endDepth = 5;
    }

    init(color, board, time) {

        super.init(color, board, time);

        this.color = color;

        this.opponentColor =
            color === "R" ? "Y" : "R";

        this.size = board.length;

        this.totalMoves =
            2 * this.size * (this.size + 1);

        this.initialTime = time;
    }

    // =====================================================
    // MAIN
    // =====================================================

    compute(board, timeRemaining) {

        if (!board || board.length === 0)
            return [0,0,0];

        this.startTime = Date.now();

        // MÁS CONSERVADOR
        this.timeLimit =
            Math.min(
                timeRemaining * 0.72,
                3.8
            );

        this.size = board.length;

        let moves =
            this.boardOps.valid_moves(board);

        if (!moves || moves.length === 0)
            return [0,0,0];

        this.movesDone =
            this.totalMoves - moves.length;

        this.configureStrategy(
            timeRemaining,
            moves.length
        );

        let selected = null;

        let randomLimit =
            Math.floor(
                this.totalMoves *
                this.openingPercent
            );

        let minimaxStart =
            Math.floor(
                this.totalMoves *
                this.minimaxStartPercent
            );

        // =====================================================
        // OPENING
        // =====================================================

        if (this.movesDone < randomLimit) {

            selected =
                this.safeRandomMove(
                    board,
                    moves
                );
        }

        // =====================================================
        // MIDGAME
        // =====================================================

        else if (
            this.movesDone < minimaxStart
        ) {

            let forced =
                this.findImmediateWin(
                    board,
                    moves
                );

            if (forced)
                selected = forced;

            else
                selected =
                    this.findBestMidMove(
                        board,
                        moves
                    );
        }

        // =====================================================
        // MINIMAX
        // =====================================================

        else {

            let dynamic =
                this.getDynamicDepth(
                    moves.length
                );

            selected =
                this.minimaxMove(
                    board,
                    moves,
                    dynamic.base,
                    dynamic.end
                );
        }

        if (
            selected &&
            this.isMoveValid(board, selected)
        )
            return selected;

        return moves[0];
    }

    // =====================================================
    // CONFIG
    // =====================================================

    configureStrategy(
        timeRemaining,
        movesLeft
    ) {

        this.FAST_MODE = false;
        this.ULTRA_FAST = false;
        this.BLITZ_20 = false;

        // =====================================
        // 10x10
        // =====================================

        if (this.size <= 10) {

            this.openingPercent = 0.01;
            this.minimaxStartPercent = 0.48;
            this.baseDepth = 5;
            this.endDepth = 7;

            if (this.initialTime <= 5) {

                this.baseDepth = 6;
                this.endDepth = 10;

                this.minimaxStartPercent = 0.5;
            }
        }

        // =====================================
        // 15x15
        // =====================================

        else if (this.size <= 15) {

            this.openingPercent = 0.02;

            this.minimaxStartPercent = 0.60;

            this.baseDepth = 4;
            this.endDepth = 6;
        }

        // =====================================
        // 20x20
        // =====================================

        else if (this.size <= 20) {

            this.openingPercent = 0.03;

            this.minimaxStartPercent = 0.74;

            this.baseDepth = 3;
            this.endDepth = 5;

            if (this.initialTime <= 5) {

                this.BLITZ_20 = true;

                this.openingPercent = 0.04;

                this.minimaxStartPercent = 0.80;

                // MÁS LIGERO
                this.baseDepth = 2;
                this.endDepth = 4;
            }
        }

        // =====================================
        // GRANDES
        // =====================================

        else {

            this.FAST_MODE = true;

            this.openingPercent = 0.05;

            this.minimaxStartPercent = 0.82;

            this.baseDepth = 2;
            this.endDepth = 4;
        }

        // =====================================
        // TIEMPO RESTANTE
        // =====================================

        if (timeRemaining < 4) {

            this.FAST_MODE = true;

            this.baseDepth =
                Math.max(
                    2,
                    this.baseDepth - 1
                );
        }

        if (timeRemaining < 2) {

            this.ULTRA_FAST = true;

            this.baseDepth = 2;

            this.endDepth = 3;
        }

        // =====================================
        // ENDGAME
        // =====================================

        if (movesLeft < 14) {

            this.endDepth += 1;
        }

        if (movesLeft < 8) {

            this.endDepth += 1;
        }
    }

    // =====================================================
    // TIME
    // =====================================================

    outOfTime() {

        return (
            (Date.now() - this.startTime)
            / 1000
        ) >= this.timeLimit;
    }

    // =====================================================
    // DEPTH
    // =====================================================

    getDynamicDepth(movesLeft) {

        let base = this.baseDepth;

        let end = this.endDepth;

        if (movesLeft < 6)
            end += 1;

        return { base, end };
    }

    // =====================================================
    // VALIDATION
    // =====================================================

    isMoveValid(board, move) {

        if (!move) return false;

        let [r,c,s] = move;

        if (
            r < 0 ||
            c < 0 ||
            r >= this.size ||
            c >= this.size
        )
            return false;

        return this.boardOps.check(
            board,
            r,
            c,
            s
        );
    }

    // =====================================================
    // UTILS
    // =====================================================

    countLines(cell) {

        return (
            ((cell&1)?1:0)+
            ((cell&2)?1:0)+
            ((cell&4)?1:0)+
            ((cell&8)?1:0)
        );
    }

    // =====================================================
    // RANDOM
    // =====================================================

    safeRandomMove(board, moves) {

        let safe = [];

        for (let move of moves) {

            if (
                !this.createsThirdLine(
                    board,
                    move
                )
            ) {

                safe.push(move);
            }
        }

        if (safe.length === 0)
            safe = moves;

        return safe[
            Math.floor(
                Math.random() * safe.length
            )
        ];
    }

    // =====================================================
    // FAST SQUARE COUNT
    // =====================================================

    countSquaresCompleted(board, move) {

        let [r,c,s] = move;

        let completed = 0;

        let cell = board[r][c];

        if (cell >= 0) {

            let next = cell | (1<<s);

            if (
                this.countLines(next) === 4
            )
                completed++;
        }

        let nr=r,nc=c,os;

        if(s===0){nr=r-1;nc=c;os=2;}
        else if(s===1){nr=r;nc=c+1;os=3;}
        else if(s===2){nr=r+1;nc=c;os=0;}
        else {nr=r;nc=c-1;os=1;}

        if (
            nr>=0 &&
            nc>=0 &&
            nr<this.size &&
            nc<this.size
        ) {

            let other = board[nr][nc];

            if (other >= 0) {

                let nextOther =
                    other | (1<<os);

                if (
                    this.countLines(nextOther)
                    === 4
                )
                    completed++;
            }
        }

        return completed;
    }

    // =====================================================
    // APPLY / UNDO
    // =====================================================

    applyMove(board, move) {

        let [r,c,s] = move;

        let changes = [];

        changes.push([r,c,board[r][c]]);

        board[r][c] |= (1<<s);

        let nr=r,nc=c,os;

        if(s===0){nr=r-1;nc=c;os=2;}
        else if(s===1){nr=r;nc=c+1;os=3;}
        else if(s===2){nr=r+1;nc=c;os=0;}
        else {nr=r;nc=c-1;os=1;}

        if (
            nr>=0 &&
            nc>=0 &&
            nr<this.size &&
            nc<this.size
        ) {

            changes.push([
                nr,
                nc,
                board[nr][nc]
            ]);

            board[nr][nc] |= (1<<os);
        }

        return changes;
    }

    undoMove(board, changes) {

        for (let c of changes) {

            board[c[0]][c[1]] = c[2];
        }
    }

    // =====================================================
    // HEURISTIC
    // =====================================================

    findBestMidMove(board, moves) {

        let ordered =
            this.orderMoves(board, moves);

        return ordered[0];
    }

    evaluateMove(board, move) {

        let [r,c] = move;

        let score = 0;

        let gain =
            this.countSquaresCompleted(
                board,
                move
            );

        score += gain * 4500;

        if (
            this.createsThirdLine(
                board,
                move
            )
        ) {

            score -= 900;
        }

        score +=
            this.chainPotential(
                board,
                r,
                c
            ) * 60;

        score +=
            this.safeZoneBonus(
                board,
                r,
                c
            ) * 25;

        return score;
    }

    chainPotential(board,r,c){

        let total = 0;

        let dirs = [
            [1,0],
            [-1,0],
            [0,1],
            [0,-1]
        ];

        for(let d of dirs){

            let nr=r+d[0];
            let nc=c+d[1];

            if(
                nr>=0 &&
                nc>=0 &&
                nr<this.size &&
                nc<this.size
            ){

                let cell=board[nr][nc];

                if(cell>=0){

                    let l=this.countLines(cell);

                    if(l===2)
                        total += 3;

                    else if(l===1)
                        total += 1;
                }
            }
        }

        return total;
    }

    safeZoneBonus(board,r,c){

        let total = 0;

        for(
            let i=Math.max(0,r-1);
            i<=Math.min(this.size-1,r+1);
            i++
        ){

            for(
                let j=Math.max(0,c-1);
                j<=Math.min(this.size-1,c+1);
                j++
            ){

                let cell = board[i][j];

                if(
                    cell>=0 &&
                    this.countLines(cell)<=1
                )
                    total++;
            }
        }

        return total;
    }

    createsThirdLine(board, move) {

        let [r,c,s] = move;

        let cell = board[r][c];

        let next = cell | (1<<s);

        return (
            this.countLines(cell) === 2 &&
            this.countLines(next) === 3
        );
    }

    // =====================================================
    // ORDER
    // =====================================================

    orderMoves(board, moves) {

        let scored = [];

        let limit = moves.length;

        // MÁS LIGERO 20x20

        if (
            this.BLITZ_20 &&
            moves.length > 45
        ) {

            limit = 22;
        }

        for(let i=0;i<limit;i++){

            scored.push({

                move: moves[i],

                score:
                    this.evaluateMove(
                        board,
                        moves[i]
                    )
            });
        }

        scored.sort(
            (a,b)=>b.score-a.score
        );

        return scored.map(x=>x.move);
    }

    // =====================================================
    // MINIMAX
    // =====================================================

    minimaxMove(
        board,
        moves,
        baseDepth,
        endDepth
    ) {

        let ordered =
            this.orderMoves(board, moves);

        let depth =
            ordered.length <= 8
            ? endDepth
            : baseDepth;

        let width;

        if (this.ULTRA_FAST)
            width = 2;

        else if (this.BLITZ_20)
            width = 3;

        else if (this.FAST_MODE)
            width = 5;

        else
            width = 6;

        let bestMove = ordered[0];

        let bestVal = -Infinity;

        for(
            let i=0;
            i<Math.min(width,ordered.length);
            i++
        ){

            if(this.outOfTime())
                break;

            let move = ordered[i];

            let gain =
                this.countSquaresCompleted(
                    board,
                    move
                );

            let changes =
                this.applyMove(
                    board,
                    move
                );

            let val =
                gain > 0
                ? gain * 260 +
                    this.minimaxLite(
                        board,
                        depth-1,
                        -Infinity,
                        Infinity,
                        true
                    )
                : this.minimaxLite(
                        board,
                        depth-1,
                        -Infinity,
                        Infinity,
                        false
                    );

            this.undoMove(board, changes);

            if(val > bestVal){

                bestVal = val;
                bestMove = move;
            }
        }

        return bestMove;
    }

    minimaxLite(
        board,
        depth,
        alpha,
        beta,
        maximizing
    ) {

        if (
            depth <= 0 ||
            this.outOfTime()
        )
            return this.fastEvaluate(board);

        let moves =
            this.boardOps.valid_moves(board);

        if (moves.length === 0)
            return this.fastEvaluate(board);

        let ordered =
            this.orderMoves(board, moves);

        let width;

        if (this.ULTRA_FAST)
            width = 1;

        else if (this.BLITZ_20)
            width = 2;

        else if (moves.length < 8)
            width = 4;

        else
            width = 3;

        if(maximizing){

            let best = -Infinity;

            for(
                let i=0;
                i<Math.min(width,ordered.length);
                i++
            ){

                if(this.outOfTime())
                    break;

                let move = ordered[i];

                let gain =
                    this.countSquaresCompleted(
                        board,
                        move
                    );

                let changes =
                    this.applyMove(
                        board,
                        move
                    );

                let val =
                    gain > 0
                    ? gain * 220 +
                        this.minimaxLite(
                            board,
                            depth-1,
                            alpha,
                            beta,
                            true
                        )
                    : this.minimaxLite(
                            board,
                            depth-1,
                            alpha,
                            beta,
                            false
                        );

                this.undoMove(board, changes);

                best = Math.max(best,val);

                alpha = Math.max(alpha,best);

                if(beta <= alpha)
                    break;
            }

            return best;
        }

        let best = Infinity;

        for(
            let i=0;
            i<Math.min(width,ordered.length);
            i++
        ){

            if(this.outOfTime())
                break;

            let move = ordered[i];

            let gain =
                this.countSquaresCompleted(
                    board,
                    move
                );

            let changes =
                this.applyMove(
                    board,
                    move
                );

            let val =
                gain > 0
                ? -gain * 220 +
                    this.minimaxLite(
                        board,
                        depth-1,
                        alpha,
                        beta,
                        false
                    )
                : this.minimaxLite(
                        board,
                        depth-1,
                        alpha,
                        beta,
                        true
                    );

            this.undoMove(board, changes);

            best = Math.min(best,val);

            beta = Math.min(beta,best);

            if(beta <= alpha)
                break;
        }

        return best;
    }

    // =====================================================
    // FAST EVAL
    // =====================================================

    fastEvaluate(board) {

        let my = 0;
        let opp = 0;

        for(let i=0;i<this.size;i++){

            for(let j=0;j<this.size;j++){

                if(board[i][j] === -1)
                    my++;

                else if(board[i][j] === -2)
                    opp++;
            }
        }

        return (my - opp) * 160;
    }

    // =====================================================
    // WIN
    // =====================================================

    findImmediateWin(board, moves) {

        let best = null;

        let bestGain = 0;

        for(let move of moves){

            let gain =
                this.countSquaresCompleted(
                    board,
                    move
                );

            if(gain > bestGain){

                bestGain = gain;

                best = move;
            }
        }

        return best;
    }
}