#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue as q
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game
        
    listNextStates : list
        List of next possible states for the current player.

    listVisitedStates : list
        List of all visited states during A*.

    listVisitedSituations : list
        List of visited game situations (state + color) for minimax/alpha-beta pruning.

    pathToTarget : list
        Sequence of states from the initial state to the target (used by A*).

    depthMax : int
        Maximum search depth for minimax/alpha-beta searches.

    dictPath : dict
        Dictionary used to reconstruct the path in A* search.

    Methods:
    --------
    copyState(state) -> list
        Returns a deep copy of the given state.

    isVisitedSituation(color, mystate) -> bool
        Checks whether a given state with a specific color has already been visited.

    getListNextStatesW(myState) -> list
        Returns a list of possible next states for the white pieces.

    getListNextStatesB(myState) -> list
        Returns a list of possible next states for the black pieces.

    isSameState(a, b) -> bool
        Checks whether two states represent the same board configuration.

    isVisited(mystate) -> bool
        Checks if a given state has been visited in search algorithms.

    getCurrentState() -> list
        Returns the combined state of both white and black pieces.

    isCheckMate(mystate) -> bool
        Determines if a state represents a checkmate configuration.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    h(state) -> int       
        Heuristic function for A* search.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8;
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    # Function for check mate for exercise 1 (white king is missing)
    def isCheckMate(self, mystate):
        
        # list of possible check mate states
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False   

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeState(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

    def h(self, state):

        # Mirem quin és el rook o King 
        # i els definim
        if state[0][2] == 2:
            kingPosition = state[1]
            rookPosition = state[0]
        else:
            kingPosition = state[0]
            rookPosition = state[1]

        # Example heuristic assusiming the target position for the king is (2,4).

        # Calculate the Manhattan distance for the king to reach the target configuration (2,4)
        rowDiff = abs(kingPosition[0] - 2)
        colDiff = abs(kingPosition[1] - 4)
        # The minimum of row and column differences corresponds to diagonal moves,
        # and the absolute difference corresponds to remaining straight moves
        hKing = min(rowDiff, colDiff) + abs(rowDiff - colDiff)

        # Heuristic for the rook, with three different cases
        if rookPosition[0] == 0 and (rookPosition[1] < 3 or rookPosition[1] > 5):
            hRook = 0
        elif rookPosition[0] != 0 and 3 <= rookPosition[1] <= 5:
            hRook = 2
        else:
            hRook = 1

        # Total heuristic is the sum of king 3and rook heuristics
        return hKing + hRook

    def changeState(self, start, to):
        # Determine which piece has moved from the start state to the next state
        if start[0] == to[0]:
            movedPieceStart = 1
            movedPieceTo = 1
        elif start[0] == to[1]:
            movedPieceStart = 1
            movedPieceTo = 0
        elif start[1] == to[0]:
            movedPieceStart = 0
            movedPieceTo = 1
        else:
            movedPieceStart = 0
            movedPieceTo = 0

        # Move the piece that changed
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])       

    def DepthFirstSearch(self, currentState, depth):
        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all nodes, we remove it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    # in the state 'son', the first piece is the one just moved
                    # We check which piece in currentState matches the one moved
                    if son[0][2] == currentState[0][2]:
                        movedPieceIndex = 0
                    else:
                        movedPieceIndex = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[movedPieceIndex], son[0])
                    # We call the method again with 'son', increasing depth
                    if self.DepthFirstSearch(son, depth + 1):
                        # If the method returns True, this means that there has
                        # been a checkmate
                        # We add the state to the pathToTarget
                        self.pathToTarget.insert(0, currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0], currentState[movedPieceIndex])

        # We remove the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)


    def worthExploring(self, state, depth):
        # First of all, check that the depth is not bigger than depthMax
        if depth > self.depthMax:
            return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If the state has been visited at a larger depth,
                # we are interested in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # Update the depth associated with the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # If never visited, add it to the dictionary at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True


    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son, depth + 1):
                # in state 'son', the first piece is the one just moved
                # we check which piece of currentState matches the one just moved
                if son[0][2] == currentState[0][2]:
                    movedPieceIndex = 0
                else:
                    movedPieceIndex = 1

                # move the piece to the new position
                self.chess.moveSim(currentState[movedPieceIndex], son[0])
                # recursive call with increased depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns True, this means there was a checkmate
                    # add the state to the pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # restore the board to its previous state
                self.chess.moveSim(son[0], currentState[movedPieceIndex])


    def BreadthFirstSearch(self, currentState, depth):
        """
        Checkmate from currentStateW
        """
        BFSQueue = q.Queue()
        # The root node has no parent, thus we add None, and -1 as the parent's depth
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there are no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Get the next node
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If not the root node, move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)

            if self.isCheckMate(node):
                # If it is checkmate, reconstruct the optimal path found
                self.reconstructPath(node, depthNode)
                break

            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode



     # TODO
        #g(n)

    # TODO
        #h(n)


     # TODO
     # A*
     # 
     #
     #        


    '''
    # frontera
    # (Valor Heuristica, Posició W) #h(), current state
    # h(n) aqui fa current stATE
        # 

        

    #RECORDA
        STATE = x , y , peça


        Donat que evaluated ha de donarnos l'element amb 
        més prioitat, importarem priority queue
    '''
    def AStarSearch(self, currentState):
        # objectiu, B-rey
        objectiu = [0, 5, 6]

        # frontera: PriorityQueue amb (f, g, estat)
        frontera = queue.PriorityQueue()
        frontera.put((self.h(currentState), 0, currentState))

        # millor cost trobat per cada node
        evaluated = {str(currentState): 0}

        # per reconstruir el camí
        
        self.dictPath = {str(currentState): (None, 0)}

        while not frontera.empty():
            f_current, g_current, current = frontera.get()
            print(f"Expanding node: {current}, f={f_current}, g={g_current}")

            if self.isCheckMate(current):
                #El g current al final sera la depth
                print("Goal found!")
                return self.reconstructPath(current, g_current)

            # Processar veïns
            for move in self.getListNextStatesW(current):
                new_g = g_current + 1
                old_g = evaluated.get(str(move), float("inf"))

                if new_g < old_g:
                    print(f"  Adding move: {move}, g={new_g}")
                    evaluated[str(move)] = new_g
                    self.dictPath[str(move)] = (current, new_g)  # guardem el pare

                    new_h = self.h(move)
                    new_f = new_g + new_h

                    frontera.put((new_f, new_g, move))

        # Si no hi ha solució
        return None


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))

    # Load initial positions of the pieces
    # White pieces
    TA[7][0] = 2  
    TA[7][5] = 6   
    TA[0][4] = 12  

    # Initialize AI chess with the board
    print("Starting AI chess...")
    aichess = Aichess(TA, True) # Al posar true indiquem que tenim un nit personalitzat

    # Print initial board
    print("Printing board:")
    aichess.chess.boardSim.print_board()

    # Get a copy of the current white state
    currentState = aichess.chess.board.currentStateW.copy()
    print("Current State:", currentState, "\n")

    # Run A* search
    #aichess.BreadthFirstSearch(currentState, depth=7)
    aichess.AStarSearch(currentState)
    print("#A* move sequence:", aichess.pathToTarget)
    print("A* End\n")
    print("Printing final board after A*:")
    aichess.chess.boardSim.print_board()
