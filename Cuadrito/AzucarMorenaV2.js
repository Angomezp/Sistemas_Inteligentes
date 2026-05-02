class AzucarMorenaV2 extends Agent {

    constructor() {
        super();
        this.boardOps = new Board();

        this.size = 0;
        this.color = "R";
        this.opponentColor = "Y";

        this.totalMoves = 0;
        this.movesDone = 0;

        this.startTime = 0;
        this.timeLimit = 0;
    }

    init(color, board, time) {
        super.init(color, board, time);

        this.color = color;
        this.opponentColor = (color === "R") ? "Y" : "R";

        this.size = board.length;
        this.totalMoves = 2 * this.size * (this.size + 1);
    }

    compute(board, timeRemaining) {

        if (!board || board.length === 0) return [0,0,0];

        this.startTime = Date.now();
        this.timeLimit = Math.min(timeRemaining * 0.85, 4.8);

        this.size = board.length;

        let moves = this.boardOps.valid_moves(board);
        if (!moves || moves.length === 0) return [0,0,0];

        this.movesDone = this.totalMoves - moves.length;

        let movesLeft = moves.length;

        // =====================================================
        // 🎯 CONFIG DINÁMICA POR TAMAÑO + TIEMPO
        // =====================================================

        let openingPercent;
        let minimaxStartPercent;

        let BASE_DEPTH;
        let END_DEPTH;

        // 🔵 TABLEROS PEQUEÑOS
        if (this.size <= 10) {
            openingPercent = 0.01;        // casi nada random
            minimaxStartPercent = 0.40;   // minimax temprano

            BASE_DEPTH = 6;
            END_DEPTH = 8;

            // 🔥 si hay buen tiempo → aún menos random
            if (timeRemaining > 8) {
                openingPercent *= 0.5;
                minimaxStartPercent -= 0.05;
            }
        }

        // 🟡 TABLEROS MEDIOS
        else if (this.size <= 20) {
            openingPercent = 0.03;
            minimaxStartPercent = 0.55;

            BASE_DEPTH = 5;
            END_DEPTH = 7;

            if (timeRemaining > 10) {
                openingPercent *= 0.7;
                minimaxStartPercent -= 0.05;
            }
        }

        // 🔴 TABLEROS GRANDES
        else {
            openingPercent = 0.05;
            minimaxStartPercent = 0.70;

            BASE_DEPTH = 4;
            END_DEPTH = 6;
        }

        // 🔥 AJUSTE FINAL POR TIEMPO BAJO
        if (timeRemaining < 3) {
            openingPercent += 0.02;
            minimaxStartPercent += 0.05;
        }

        let randomLimit = Math.floor(this.totalMoves * openingPercent);
        let minimaxStart = Math.floor(this.totalMoves * minimaxStartPercent);

        let selected = null;

        // =====================================================
        // FASES
        // =====================================================

        // 🔹 OPENING
        if (this.movesDone < randomLimit) {
            selected = moves[Math.floor(Math.random() * moves.length)];
        }

        // 🔴 MINIMAX
        else if (this.movesDone >= minimaxStart) {

            let dynamic = this.getDynamicDepth(
                timeRemaining,
                BASE_DEPTH,
                END_DEPTH,
                movesLeft
            );

            selected = this.minimaxMove(
                board,
                moves,
                dynamic.base,
                dynamic.end
            );
        }

        // 🟡 MIDGAME
        else {
            let win = this.findImmediateWin(board, moves);
            if (win) selected = win;
            else selected = this.findBestMidMove(board, moves);
        }

        if (this.isMoveValid(board, selected)) return selected;

        for (let m of moves)
            if (this.isMoveValid(board, m)) return m;

        return moves[0];
    }

    // =====================================================
    // CONTROL TIEMPO
    // =====================================================

    outOfTime() {
        return ((Date.now() - this.startTime) / 1000) >= this.timeLimit;
    }

    // =====================================================
    // DEPTH ADAPTATIVO
    // =====================================================

    getDynamicDepth(time, base, end, movesLeft) {

        if (time > 12) { base += 1; end += 1; }
        else if (time < 5) { base -= 1; end -= 1; }

        // 🔥 control endgame
        if (movesLeft < 12) {
            base -= 1;
            end -= 1;
        }

        if (movesLeft < 6) {
            base -= 2;
            end -= 2;
        }

        return {
            base: Math.max(2, base),
            end: Math.max(3, end)
        };
    }

    // =====================================================
    // VALIDACIÓN
    // =====================================================

    isMoveValid(board, move) {
        if (!move || move.length !== 3) return false;

        let [r,c,s] = move;

        if (r<0||c<0||r>=this.size||c>=this.size||s<0||s>3)
            return false;

        try {
            return this.boardOps.check(board, r, c, s);
        } catch(e) {
            return false;
        }
    }

    // =====================================================
    // UTILIDADES
    // =====================================================

    countLines(cell){
        return ((cell&1)?1:0)+((cell&2)?1:0)+((cell&4)?1:0)+((cell&8)?1:0);
    }

    countPlayerSquares(board, color){
        let target = color==="R"?-1:-2, total=0;
        for(let i=0;i<this.size;i++)
            for(let j=0;j<this.size;j++)
                if(board[i][j]===target) total++;
        return total;
    }

    applyMove(board, move, color){
        if(!this.isMoveValid(board, move)) return false;
        let [r,c,s]=move;
        let mark=color==="R"?-1:-2;
        return this.boardOps.move(board,r,c,s,mark);
    }

    applyMoveAndGetSquares(board, move, color){
        let before=this.countPlayerSquares(board,color);
        this.applyMove(board,move,color);
        let after=this.countPlayerSquares(board,color);
        return after-before;
    }

    // =====================================================
    // MIDGAME
    // =====================================================

    findBestMidMove(board, moves){
        let best=moves[0], bestScore=-Infinity;

        for(let move of moves){
            if(!this.isMoveValid(board,move)) continue;

            let score=this.evaluateMidMove(board,move);

            if(score>bestScore){
                bestScore=score;
                best=move;
            }
        }

        return best;
    }

    evaluateMidMove(board, move){
        let [r,c]=move;
        let score=0;

        let gain = this.countSquaresCompleted(board,move);
        score += gain * 1400;

        if(this.createsThirdLine(board,move))
            score -= 300;

        score += this.chainPotential(board,r,c) * 35;
        score += this.safeZoneBonus(board,r,c) * 15;

        return score;
    }

    chainPotential(board,r,c){
        let total=0;
        let dirs=[[1,0],[-1,0],[0,1],[0,-1]];
        for(let d of dirs){
            let nr=r+d[0],nc=c+d[1];
            if(nr>=0&&nc>=0&&nr<this.size&&nc<this.size){
                let cell=board[nr][nc];
                if(cell>=0){
                    let l=this.countLines(cell);
                    if(l===2) total+=2;
                    else if(l===1) total++;
                }
            }
        }
        return total;
    }

    safeZoneBonus(board,r,c){
        let total=0;
        let dirs=[[1,0],[-1,0],[0,1],[0,-1]];
        for(let d of dirs){
            let nr=r+d[0],nc=c+d[1];
            if(nr>=0&&nc>=0&&nr<this.size&&nc<this.size){
                let cell=board[nr][nc];
                if(cell>=0 && this.countLines(cell)<=1) total++;
            }
        }
        return total;
    }

    // =====================================================
    // MINIMAX (seguro)
    // =====================================================

    minimaxMove(board, moves, baseDepth, endDepth){

        let ordered = this.orderMoves(board, moves);

        let depth = ordered.length <= 6 ? endDepth : baseDepth;

        let maxRoot = ordered.length <= 10 ? ordered.length : 8;

        let bestMove = ordered[0];
        let bestVal = -Infinity;

        for(let i=0;i<Math.min(maxRoot, ordered.length);i++){

            if (this.outOfTime()) break;

            let move=ordered[i];
            if(!this.isMoveValid(board,move)) continue;

            let temp=this.boardOps.clone(board);

            let gain=this.applyMoveAndGetSquares(temp,move,this.color);

            let val =
                gain>0
                ? gain*90 + this.minimaxLite(temp,depth-1,-Infinity,Infinity,true)
                : this.minimaxLite(temp,depth-1,-Infinity,Infinity,false);

            if(val>bestVal){
                bestVal=val;
                bestMove=move;
            }
        }

        return bestMove;
    }

    minimaxLite(board, depth, alpha, beta, maximizing){

        if (this.outOfTime()) return this.fastEvaluate(board);

        let moves=this.boardOps.valid_moves(board);

        if(depth<=0 || moves.length===0)
            return this.fastEvaluate(board);

        let width;

        if (moves.length < 6) width = moves.length;
        else if (moves.length < 12) width = 4;
        else width = 3;

        let color=maximizing?this.color:this.opponentColor;

        if(maximizing){

            let best=-Infinity;

            for(let i=0;i<Math.min(width,moves.length);i++){

                if (this.outOfTime()) break;

                let move=moves[i];
                if(!this.isMoveValid(board,move)) continue;

                let temp=this.boardOps.clone(board);

                let gain=this.applyMoveAndGetSquares(temp,move,color);

                let val =
                    gain>0
                    ? gain*70 + this.minimaxLite(temp,depth-1,alpha,beta,true)
                    : this.minimaxLite(temp,depth-1,alpha,beta,false);

                best=Math.max(best,val);
                alpha=Math.max(alpha,best);
                if(beta<=alpha) break;
            }

            return best;
        }

        let best=Infinity;

        for(let i=0;i<Math.min(width,moves.length);i++){

            if (this.outOfTime()) break;

            let move=moves[i];
            if(!this.isMoveValid(board,move)) continue;

            let temp=this.boardOps.clone(board);

            let gain=this.applyMoveAndGetSquares(temp,move,color);

            let val =
                gain>0
                ? -gain*70 + this.minimaxLite(temp,depth-1,alpha,beta,false)
                : this.minimaxLite(temp,depth-1,alpha,beta,true);

            best=Math.min(best,val);
            beta=Math.min(beta,best);
            if(beta<=alpha) break;
        }

        return best;
    }

    orderMoves(board,moves){
        return moves.slice().sort((a,b)=>
            this.evaluateMidMove(board,b) -
            this.evaluateMidMove(board,a)
        );
    }

    fastEvaluate(board){
        let my=this.countPlayerSquares(board,this.color);
        let opp=this.countPlayerSquares(board,this.opponentColor);
        return (my-opp)*140;
    }

    createsThirdLine(board,move){
        if(!this.isMoveValid(board,move)) return false;
        let [r,c,s]=move;
        let cell=board[r][c];
        let next=cell|(1<<s);
        return this.countLines(cell)===2 && this.countLines(next)===3;
    }

    countSquaresCompleted(board,move){
        if(!this.isMoveValid(board,move)) return 0;
        let temp=this.boardOps.clone(board);
        let before=this.countPlayerSquares(temp,this.color);
        this.applyMove(temp,move,this.color);
        let after=this.countPlayerSquares(temp,this.color);
        return after-before;
    }

    findImmediateWin(board,moves){
        let best=null, bestGain=0;
        for(let move of moves){
            if(!this.isMoveValid(board,move)) continue;
            let gain=this.countSquaresCompleted(board,move);
            if(gain>bestGain){
                bestGain=gain;
                best=move;
            }
        }
        return best;
    }
}