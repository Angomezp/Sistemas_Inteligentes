/*
 * AzucarMorena ULTRA ADAPTIVE FINAL
 *
 * ✔ Optimizado por tamaño de tablero
 * ✔ Control dinámico por tiempo
 * ✔ Minimax adaptativo (profundidad + anchura)
 * ✔ Estrategia intermedia fuerte pero ligera
 * ✔ Validación segura
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
    }

    init(color, board, time) {
        super.init(color, board, time);

        this.color = color;
        this.opponentColor = (color === "R") ? "Y" : "R";

        this.size = board.length;
        this.totalMoves = 2 * this.size * (this.size + 1);
    }

    // =====================================================
    // MAIN
    // =====================================================

    compute(board, timeRemaining) {

        if (!board || board.length === 0) return [0,0,0];

        this.size = board.length;

        let moves = this.boardOps.valid_moves(board);
        if (!moves || moves.length === 0) return [0,0,0];

        this.movesDone = this.totalMoves - moves.length;

        // =====================================================
        // CONFIG POR TAMAÑO
        // =====================================================

        let OPENING_PERCENT, MINIMAX_START, BASE_DEPTH, END_DEPTH, EXTRA_MID = false;

        if (this.size <= 10) {
            OPENING_PERCENT = 0.02;
            MINIMAX_START = 0.45;
            BASE_DEPTH = 6;
            END_DEPTH = 8;
        }
        else if (this.size <= 15) {
            OPENING_PERCENT = 0.02;
            MINIMAX_START = 0.50;
            BASE_DEPTH = 5;
            END_DEPTH = 7;
        }
        else if (this.size <= 20) {
            OPENING_PERCENT = 0.03;
            MINIMAX_START = 0.60;
            BASE_DEPTH = 4;
            END_DEPTH = 6;
        }
        else if (this.size <= 28) {
            OPENING_PERCENT = 0.04;
            MINIMAX_START = 0.65;
            BASE_DEPTH = 4;
            END_DEPTH = 5;
        }
        else {
            OPENING_PERCENT = 0.05;
            MINIMAX_START = 0.75;
            BASE_DEPTH = 3;
            END_DEPTH = 4;
            EXTRA_MID = true;
        }

        let randomLimit = Math.floor(this.totalMoves * OPENING_PERCENT);
        let minimaxStart = Math.floor(this.totalMoves * MINIMAX_START);

        let selected = null;

        // =====================================================
        // FASES
        // =====================================================

        if (this.movesDone < randomLimit) {
            selected = moves[Math.floor(Math.random() * moves.length)];
        }

        else if (this.movesDone >= minimaxStart) {

            let dynamic = this.getDynamicDepth(
                timeRemaining,
                BASE_DEPTH,
                END_DEPTH,
                moves.length
            );

            selected = this.minimaxMove(
                board,
                moves,
                dynamic.base,
                dynamic.end,
                timeRemaining
            );
        }

        else {
            let win = this.findImmediateWin(board, moves);
            if (win) selected = win;
            else selected = this.findBestMidMove(board, moves, EXTRA_MID);
        }

        if (this.isMoveValid(board, selected)) return selected;

        for (let m of moves)
            if (this.isMoveValid(board, m)) return m;

        return moves[0];
    }

    // =====================================================
    // DINÁMICA TIEMPO
    // =====================================================

    getDynamicDepth(time, base, end, movesLeft) {

        if (time > 20) {
            base += 2; end += 2;
        }
        else if (time > 10) {
            base += 1; end += 1;
        }
        else if (time < 5) {
            base -= 1; end -= 1;
        }
        else if (time < 2) {
            base -= 2; end -= 2;
        }

        if (movesLeft < 12) {
            base += 1;
            end += 2;
        }

        return {
            base: Math.max(2, base),
            end: Math.max(3, end)
        };
    }

    getDynamicWidth(time) {
        if (time > 15) return 5;
        if (time > 8) return 4;
        if (time > 3) return 3;
        return 2;
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

    findBestMidMove(board, moves, extra){

        let best=moves[0], bestScore=-Infinity;

        for(let move of moves){

            if(!this.isMoveValid(board,move)) continue;

            let score=this.evaluateMidMove(board,move,extra);

            if(score>bestScore){
                bestScore=score;
                best=move;
            }
        }

        return best;
    }

    evaluateMidMove(board, move, extra){

        let [r,c]=move;
        let score=0;

        score+=this.countSquaresCompleted(board,move)*1300;

        if(this.createsThirdLine(board,move)) score-=300;

        score+=this.chainPotential(board,r,c)*35;
        score+=this.safeZoneBonus(board,r,c)*15;

        if(extra){
            score+=this.openAreaBonus(board,r,c)*30;
            score+=this.centerControl(r,c)*15;
        }

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

    openAreaBonus(board,r,c){
        let total=0;
        for(let i=Math.max(0,r-1);i<=Math.min(this.size-1,r+1);i++){
            for(let j=Math.max(0,c-1);j<=Math.min(this.size-1,c+1);j++){
                let cell=board[i][j];
                if(cell>=0 && this.countLines(cell)<=1) total++;
            }
        }
        return total;
    }

    centerControl(r,c){
        let center=(this.size-1)/2;
        return this.size-(Math.abs(r-center)+Math.abs(c-center));
    }

    // =====================================================
    // MINIMAX
    // =====================================================

    minimaxMove(board, moves, baseDepth, endDepth, timeRemaining){

        let ordered=this.orderMoves(board,moves);

        let depth=baseDepth;
        if(ordered.length<=6) depth=endDepth;

        let width=this.getDynamicWidth(timeRemaining);

        let bestMove=ordered[0];
        let bestVal=-Infinity;

        for(let i=0;i<Math.min(width,ordered.length);i++){

            let move=ordered[i];
            if(!this.isMoveValid(board,move)) continue;

            let temp=this.boardOps.clone(board);

            let gain=this.applyMoveAndGetSquares(temp,move,this.color);

            let val=
                gain>0
                ? gain*90+this.minimaxLite(temp,depth-1,-Infinity,Infinity,true,timeRemaining)
                : this.minimaxLite(temp,depth-1,-Infinity,Infinity,false,timeRemaining);

            if(val>bestVal){
                bestVal=val;
                bestMove=move;
            }
        }

        return bestMove;
    }

    minimaxLite(board, depth, alpha, beta, maximizing, time){

        let moves=this.boardOps.valid_moves(board);

        if(depth===0 || moves.length===0)
            return this.fastEvaluate(board);

        let width=this.getDynamicWidth(time);
        let color=maximizing?this.color:this.opponentColor;

        if(maximizing){

            let best=-Infinity;

            for(let i=0;i<Math.min(width,moves.length);i++){

                let move=moves[i];
                if(!this.isMoveValid(board,move)) continue;

                let temp=this.boardOps.clone(board);
                let gain=this.applyMoveAndGetSquares(temp,move,color);

                let val=
                    gain>0
                    ? gain*70+this.minimaxLite(temp,depth-1,alpha,beta,true,time)
                    : this.minimaxLite(temp,depth-1,alpha,beta,false,time);

                best=Math.max(best,val);
                alpha=Math.max(alpha,best);
                if(beta<=alpha) break;
            }

            return best;
        }

        let best=Infinity;

        for(let i=0;i<Math.min(width,moves.length);i++){

            let move=moves[i];
            if(!this.isMoveValid(board,move)) continue;

            let temp=this.boardOps.clone(board);
            let gain=this.applyMoveAndGetSquares(temp,move,color);

            let val=
                gain>0
                ? -gain*70+this.minimaxLite(temp,depth-1,alpha,beta,false,time)
                : this.minimaxLite(temp,depth-1,alpha,beta,true,time);

            best=Math.min(best,val);
            beta=Math.min(beta,best);
            if(beta<=alpha) break;
        }

        return best;
    }

    orderMoves(board,moves){
        return moves.sort((a,b)=>
            this.evaluateMidMove(board,b,false)-
            this.evaluateMidMove(board,a,false)
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