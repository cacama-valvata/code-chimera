import pygame
import math

class Obstacle(pygame.Rect):

    TOPLEFT = 0
    TOPRIGHT = 1
    BOTTOMRIGHT = 2
    BOTTOMLEFT = 3

    def __init__(self, x, y, width, height, color=(0, 0, 255)):
        pygame.Rect.__init__(self, x, y, width, height)
        self.color = color
        self.corners = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]

    def update(self, world):
        pygame.draw.rect(world.screen, self.color, self)

    # check to see if point inside obstacle, or within radius distance of obstacle
    def collidepoint(self, point, radius=0):
        return round(point[0]) >= self.corners[Obstacle.TOPLEFT][0] - radius and round(point[1]) >= self.corners[Obstacle.TOPLEFT][1] - radius and \
                round(point[0]) <= self.corners[Obstacle.BOTTOMRIGHT][0] + radius and round(point[1]) <= self.corners[Obstacle.BOTTOMRIGHT][1] + radius

    # return nearest point of collision and side of collision on rectangle
    def collideline(self, line, radius=0):
        start = line[0], line[1]
        end = line[2], line[3]
        newCorners = [
            (self.corners[0][0] - radius, self.corners[0][1] - radius),
            (self.corners[1][0] + radius, self.corners[1][1] - radius),
            (self.corners[2][0] + radius, self.corners[2][1] + radius),
            (self.corners[3][0] - radius, self.corners[3][1] + radius)
        ]
        length = dist(start, end)
        minDist = float("inf")
        nearestPoint = None
        for i in range(4):
            point = getIntersection(start, end, newCorners[i], newCorners[(i + 1) % 4])
            if point and self.collidepoint(point, radius):
                if (point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]) >= 0:
                    distToPoint = dist(point, start)
                    if distToPoint < minDist and distToPoint <= length:
                        minDist = distToPoint
                        nearestPoint = point
        return nearestPoint
        

### https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python ####
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = float(Dx) / D
        y = float(Dy) / D
        return x,y
    else:
        return False
#####################################################################################################################

def getIntersection(p1, p2, p3, p4):
    return intersection(line(p1, p2), line(p3, p4))

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def rotateVector(v, angle):
    return math.cos(angle) * v[0] - math.sin(angle) * v[1], math.sin(angle) * v[0] + math.cos(angle) * v[1]









