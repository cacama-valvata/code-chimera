# https://permadi.com/1996/05/ray-casting-tutorial
# Parts from "Introduction" to "Texture Mapped Walls"
# Shoutout TA Connor Tsui for how to draw sky and ground (mini lecture)
# Shoutout TA Ben2 for explanation of finding walls at intersection vs
# naively stepping
# Shoutout Stephen Mao, discussed how to calculate height of a
# triangle. stmao@andrew.cmu.edu

# Spritecasting inspiration
# https://www.youtube.com/watch?v=CcNCgmVJiBY
# https://www.youtube.com/watch?v=Q4cDDNCXfHA 
# https://www.youtube.com/watch?v=kyI-Ken7aAk

from cmu_112_graphics import *
from helpers import *
import math
from operator import itemgetter

class Raycaster:
    def __init__(self, app, maze):
        self.maze = maze.maze
        (self.cellWidth, self.cellHeight) = getCellSpecs(app, self.maze)
        self.FOV = 60
        self.playerHeight = app.wallHeight/2

        self.numRays = 400
        self.angleBetweenRays = self.FOV/self.numRays
        self.baseWallColor = (189, 191, 163)
        self.baseSkyAndGroundColor = rgbString(37, 38, 38)

# Creating dictionary of sprite with dist from player for use in drawing
# priority to hide behind walls
# https://www.youtube.com/watch?v=CcNCgmVJiBY
# https://www.youtube.com/watch?v=Q4cDDNCXfHA 
# https://www.youtube.com/watch?v=kyI-Ken7aAk
    def spriteCaster(self, app):
        spriteList = []
        for sprite in app.sprites:
            screenX, screenY = sprite.getSpriteCoords(app)
            # Only including in list if in FOV for efficiency
            # Do not run line check on enemy (come back)
            if sprite.inFOV(screenX, app):
                dist = getDistance(app.player.xPos, app.player.yPos,
                sprite.xPos, sprite.yPos)
                spriteDict = {'dist': dist, 'type': 'sprite', 'obj': sprite}
                spriteList.append(spriteDict)
        # print(spriteList[0]['obj'].getSpriteCoords(app))
        return spriteList

    def drawScene(self, app, canvas):
        planeWidth = app.width
        planeHeight = app.height
        cy = planeHeight/2
        currX = 0
        xAdj = planeWidth/self.numRays

        wallsList = self.createSlices(app, canvas)
        spritesList = self.spriteCaster(app)
        allElements = wallsList + spritesList

        # sort by distance
        allElements = sorted(allElements, key=itemgetter('dist'), reverse=True)

        for element in allElements:
            if element['type'] == 'wall':
                x0 = element['xNum']*xAdj
                x1 = x0 + xAdj
                self.drawWall(app, canvas, x0, x1, cy, element['projHeight'],
            element['wallColor'])
                self.drawSky(app, canvas, x0, x1, cy, element['projHeight'])
                self.drawGround(app, canvas, x0, x1, cy, element['projHeight'])

                currX = x1

            elif element['type'] == 'sprite':
                element['obj'].redraw(app, canvas)

    def drawWall(self, app, canvas, x0, x1, cy, height, color): 
        (y0, y1) = (cy-(height/2),cy+(height/2))
        currX = x1
        canvas.create_rectangle(x0,y0,x1,y1,fill=color, 
        outline=color)

    def drawSky(self, app, canvas, x0, x1, cy, height):
        (y0, y1) = (cy-(height/2), 0)
        canvas.create_rectangle(x0,y0,x1,y1,fill=self.baseSkyAndGroundColor, 
        outline=self.baseSkyAndGroundColor)

    def drawGround(self, app, canvas, x0, x1, cy, height):
        (y0, y1) = (cy+(height/2), app.height)
        canvas.create_rectangle(x0,y0,x1,y1,fill=self.baseSkyAndGroundColor, 
        outline=self.baseSkyAndGroundColor)

    def createSlices(self, app, canvas):
        distsWithColors = self.getDists(app, canvas)
        slices = []
        distToPlane = (app.width/2)*math.tan(math.radians(30))

        for distWithColor in distsWithColors:
            # Get projection height
            # If else to avoid div by 0
            if distWithColor['dist'] > 0: 
                projHeight = ((app.wallHeight/distWithColor['dist']
                )*app.distToPlane)
            else:
                # might cause problems come back
                projHeight = ((app.wallHeight/0.0000001)*app.distToPlane)
            # Saving dist in dictionary to use in depth buffer
            sliceDict = {'dist': distWithColor['dist'], 'type': 'wall', 
            'projHeight': projHeight, 'wallColor': distWithColor['wallColor'],
            'xNum': distWithColor['xNum']}

            slices.append(sliceDict)
        return slices

    def getDists(self, app, canvas):
        distsWithColor = []
        angle = app.player.angle - 30
        for i in range(self.numRays):
            angle += self.angleBetweenRays
            # if statement here might be wrong. come back if bugs.
            if angle > 360:
                angle = 0
            ray = self.getRay(app, angle)
            # Need to get xNum to draw x position correctly with depth buffer
            ray['xNum'] = i
            distsWithColor.append(ray)
        return distsWithColor

    def getRay(self, app, angle):
        if angle > 90 and angle < 270:
            distHor = self.horizontalUpRay(app, angle)
        else:
            distHor = self.horizontalDownRay(app, angle)
        if angle > 0 and angle < 180:
            distVer = self.verticalRightRay(app, angle)
        else:
            distVer = self.verticalLeftRay(app, angle)
        # https://stackoverflow.com/questions/66644579/how-do-i-fix-warped-walls-in-my-raycaster
        fisheyeAdjuster = math.cos(math.radians(angle-app.player.angle))
        if distHor[2] < distVer[2]:
            return {'dist': distHor[2]*fisheyeAdjuster, 'wallColor': distHor[3]}
        else:
            return {'dist': distVer[2]*fisheyeAdjuster, 'wallColor': distVer[3]}

# rAdj and cAdj needed because intersection checks top most cell, causing
# errors in cases Vertical Left and Horizontal up. 
    def checkFirstIntersection(self, app, px, py, rAdj, cAdj):
        (intersectionRow, intersectionCol) = getCell(app, px, py, self.maze)
        # Only check if point on map
        if (intersectionRow >= 0 and intersectionRow < len(self.maze) and 
        intersectionCol >= 0 and intersectionCol < len(self.maze)):
            if self.maze[intersectionRow+rAdj][intersectionCol+cAdj] == 1:
                return (True, getDistance(app.player.xPos, app.player.yPos, 
                px, py))
        return (False, 0)

    def checkOtherIntersections(self, app, px, py, rAdj, cAdj):
        (intersectionRow, intersectionCol) = getCell(app, px, py, self.maze)
        if (intersectionRow >= 0 and intersectionRow < len(self.maze) and 
            intersectionCol >= 0 and intersectionCol < len(self.maze)):
            if self.maze[intersectionRow+rAdj][intersectionCol+cAdj] == 1:
                return (True, getDistance(app.player.xPos, app.player.yPos, px, 
                py))
            # If point off map, just return a giant number 
        else:
            return (True, 10000000000000)
        return (False, 0)

    def wallShadeFormula(self, px, py, app):
        (row, col) = getCell(app, px, py, app.maze.maze)
        if ((app.player.row, app.player.col) == 
        (app.enemy.row, app.enemy.col)):
            self.baseSkyAndGroundColor = rgbString(0, 0, 0)
        else:
            self.baseSkyAndGroundColor = rgbString(69, 69, 69)

        playerToSliceDist = getDistance(px, py, app.player.xPos, 
        app.player.yPos)

        # 80 is a constant chosen to make reasonable divisor
        divisor = playerToSliceDist/50
        if divisor < 1:
            divisor = 1
        # elif divisor > 12:
        #     return rgbString(0, 0, 0)                
        
        # For both divisior = 1 and divisior < 7
        wallColor = []
        for part in self.baseWallColor:
            wallColor.append(int(part/divisor))
        wallColor = rgbString(wallColor[0], wallColor[1], wallColor[2])
        return wallColor

    def verticalRightRay(self, app, angle):
    # end variable first arg: True or False condition. second arg: distance
    # value. 
        end = (False, 0)
        px = self.cellWidth*(app.player.col+1)
        py = (app.player.yPos - 
        math.tan(math.radians(angle-90))*(px-app.player.xPos))

        end = self.checkFirstIntersection(app, px, py, 0, 0)

        while end[0] != True:
            Ya = self.cellWidth*(math.tan(math.radians(angle-90)))
            px = px+self.cellWidth
            py = py-Ya
            end = self.checkOtherIntersections(app, px, py, 0, 0)

        color = self.wallShadeFormula(px, py, app)
        return (px, py, end[1], color)

    def verticalLeftRay(self, app, angle):
        end = (False, 0)
        px = self.cellWidth*(app.player.col)
        py = (app.player.yPos - 
        math.tan(math.radians(angle-90))*(px-app.player.xPos))

        end = self.checkFirstIntersection(app, px, py, 0, -1)

        while end[0] != True:
            Ya = self.cellWidth*(math.tan(math.radians(angle-90)))
            px = px-self.cellWidth
            py = py+Ya
            end = self.checkOtherIntersections(app, px, py, 0, -1)
        # Test other intersections until hit wall

        color = self.wallShadeFormula(px, py, app)
        return (px, py, end[1], color)

    def horizontalDownRay(self, app, angle):
        end = (False, 0)
        py = self.cellHeight*(app.player.row + 1)
        #0.0001 added to avoid div by 0 error
        px = (app.player.xPos + 
        (app.player.yPos - py)/(math.tan(math.radians(angle-90))
        +0.0001))

        end = self.checkFirstIntersection(app, px, py, 0, 0)

        while end[0] != True:
            Xa = self.cellHeight/((math.tan(math.radians(angle-90))
            +0.0001))
            px = px-Xa
            py = py+self.cellHeight
            end = self.checkOtherIntersections(app, px, py, 0, 0)

        color = self.wallShadeFormula(px, py, app)
        return (px, py, end[1], color)  

    def horizontalUpRay(self, app, angle):
        end = (False, 0)
        py = self.cellHeight*(app.player.row)
        px = (app.player.xPos + 
        (app.player.yPos - py)/(math.tan(math.radians(angle-90))
        +0.0001))

        end = self.checkFirstIntersection(app, px, py, -1, 0)

        # Test other intersections until hit wall
        while end[0] != True:
            Xa = (self.cellHeight/(math.tan(math.radians(angle-90))
            +0.0001))
            px = px+Xa
            py = py-self.cellHeight
            end = self.checkOtherIntersections(app, px, py, -1, 0)

        color = self.wallShadeFormula(px, py, app)
        return (px, py, end[1], color)

    def redraw(self, app, canvas):
        self.drawScene(app, canvas)