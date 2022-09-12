import cv2
import numpy as np
import math

###
### All methods in this file are taken from the echonet repository https://github.com/echonet/dynamic
### to compute comparable values for the ejection fraction ef
### Only one method, obtainContourPoints, to directly use the input masks instead of paths
###

# Change to Method of Disks
def volumeMethodOfDisks(x1, y1, x2, y2, number, lowerInterceptAveragePoints, higherInterceptAveragePoints):
    # Long axis length and perp initialzation
    distance = getDistance([x1, y1], [x2, y2])
    parallelSeperationDistance = distance / (number + 1)

    lowerInterceptAveragePoints = np.asarray(lowerInterceptAveragePoints).swapaxes(1, 0)
    higherInterceptAveragePoints = np.asarray(higherInterceptAveragePoints).swapaxes(1, 0)
    # Simpson Volume Methods
    volume = 0

    for i in range(len(lowerInterceptAveragePoints)):
        diameter = getDistance(lowerInterceptAveragePoints[i], higherInterceptAveragePoints[i])
        radius = diameter / 2
        diskVolume = math.pi * radius ** 2 * parallelSeperationDistance
        volume += diskVolume

    return volume

def obtainContourPoints(mask):
    # get contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Gets all contour points
    points = []
    for pt in contours:
        for i in pt:
            for coord in i:
                points.append(coord.tolist())
    # print(points)
    return points


# Distance Between 2 Points
def getDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def findCorrespondingMaskPoints(weighted_avg, lowerIntercept, higherIntercept, x1, y1, x2, y2, slope, i):
    # Calculate perpendicular slope
    try:
        perp_slope = -1 / slope
    except:
        perp_slope = 10000

    # Indexing
    lowerIndex = 0
    higherIndex = 0

    # Make sure its from top to bottom direction
    if (weighted_avg[-1][0] + weighted_avg[-1][1]) < (weighted_avg[0][0] + weighted_avg[0][1]):
        weighted_avg = weighted_avg[::-1]

    # Make sure its from top to bottom direction
    if getDistance(weighted_avg[0], higherIntercept[0]) > getDistance(weighted_avg[0], higherIntercept[-1]):
        higherIntercept = higherIntercept[::-1]

    # Make sure its from top to bottom direction
    if getDistance(weighted_avg[0], lowerIntercept[0]) > getDistance(weighted_avg[0], lowerIntercept[-1]):
        lowerIntercept = lowerIntercept[::-1]

    higherInterceptAveragePoints = []
    lowerInterceptAveragePoints = []

    for averagePoint in weighted_avg:
        try:
            condition = True
            count = 0
            while condition:
                higherIndex = max(higherIndex, len(higherInterceptAveragePoints))
                point = higherIntercept[higherIndex]
                if higherIndex == 0:
                    prev_point = [x1, y1] if getDistance(point, [x1, y1]) < getDistance(point, [x2, y2]) else [x2, y2]
                    start_point = prev_point[:]
                else:
                    prev_point = higherIntercept[higherIndex - 1]

                new_slope = getSlope(point, averagePoint)
                prev_slope = getSlope(prev_point, averagePoint)
                betweenCond = ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (
                        point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])) and abs(new_slope) > abs(
                    slope) and abs(prev_slope) > abs(slope)
                slopeCond = (new_slope >= perp_slope and prev_slope <= perp_slope) or (
                        new_slope <= perp_slope and prev_slope >= perp_slope)

                count += 1
                higherIndex += 1

                if perp_slope == 10000:
                    if (point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (
                            point[0] > averagePoint[0] and prev_point[0] < averagePoint[0]):
                        higherInterceptAveragePoints.append(point)
                        condition = False
                        higherIndex -= 1
                elif not (len(higherInterceptAveragePoints) > 0 and higherInterceptAveragePoints[
                    0] == point and point == start_point):
                    if slopeCond and not betweenCond:
                        higherInterceptAveragePoints.append(point)
                        condition = False
                        higherIndex -= 1
                    elif (abs(perp_slope) > 6) and (
                            (new_slope > 1.1 * abs(slope) and prev_slope < -1.1 * abs(slope)) or (
                            new_slope < -1.1 * abs(slope) and prev_slope > 1.1 * abs(slope))):
                        higherInterceptAveragePoints.append(point)
                        condition = False
                        higherIndex -= 1
                    elif (abs(slope) > 6) and ((point[1] < averagePoint[1] and prev_point[1] > averagePoint[1]) or (
                            point[1] > averagePoint[1] and prev_point[1] < averagePoint[1])):
                        higherInterceptAveragePoints.append(point)
                        condition = False
                        higherIndex -= 1
                    elif higherIndex + 1 >= len(higherIntercept):
                        higherIndex -= count
                        if higherIndex == 0:
                            higherInterceptAveragePoints.append(start_point)
                        else:
                            higherInterceptAveragePoints.append(higherIntercept[higherIndex])
                        condition = False
                        higherIndex -= 1
        except:
            higherInterceptAveragePoints.append(higherIntercept[-1])

    for averagePoint in weighted_avg:
        try:
            condition = True
            count = 0
            while condition:
                lowerIndex = max(lowerIndex, len(lowerInterceptAveragePoints))
                point = lowerIntercept[lowerIndex]

                if lowerIndex == 0:
                    prev_point = [x1, y1] if getDistance(point, [x1, y1]) < getDistance(point, [x2, y2]) else [x2, y2]
                    start_point = prev_point[:]
                else:
                    prev_point = lowerIntercept[lowerIndex - 1]

                new_slope = getSlope(point, averagePoint)
                prev_slope = getSlope(prev_point, averagePoint)
                betweenCond = ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (
                        point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])) and abs(new_slope) > abs(
                    slope) and abs(prev_slope) > abs(slope)
                slopeCond = (new_slope >= perp_slope and prev_slope <= perp_slope) or (
                        new_slope <= perp_slope and prev_slope >= perp_slope)

                count += 1
                lowerIndex += 1

                if perp_slope == 10000:
                    if ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (
                            point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])):
                        lowerInterceptAveragePoints.append(point)
                        condition = False
                        lowerIndex -= 1
                elif not (len(lowerInterceptAveragePoints) > 0 and lowerInterceptAveragePoints[
                    0] == point and point == start_point):
                    if slopeCond and not betweenCond:
                        lowerInterceptAveragePoints.append(point)
                        condition = False
                        lowerIndex -= 1
                    elif (abs(perp_slope) > 6) and (
                            (new_slope > 1.1 * abs(slope) and prev_slope < -1.1 * abs(slope)) or (
                            new_slope < -1.1 * abs(slope) and prev_slope > 1.1 * abs(slope))):
                        lowerInterceptAveragePoints.append(point)
                        condition = False
                        lowerIndex -= 1
                    elif (abs(slope) > 6) and ((point[1] < averagePoint[1] and prev_point[1] > averagePoint[1]) or (
                            point[1] > averagePoint[1] and prev_point[1] < averagePoint[1])):
                        lowerInterceptAveragePoints.append(point)
                        condition = False
                        lowerIndex -= 1
                    elif lowerIndex + 1 >= len(lowerIntercept):
                        lowerIndex -= count
                        if lowerIndex == 0:
                            lowerInterceptAveragePoints.append(start_point)
                        else:
                            lowerInterceptAveragePoints.append(lowerIntercept[lowerIndex])
                        condition = False
                        lowerIndex -= 1
        except:
            lowerInterceptAveragePoints.append(lowerIntercept[-1])

    matchedAveragePoints = [lowerInterceptAveragePoints[i] + higherInterceptAveragePoints[i] for i in
                            range(len(lowerInterceptAveragePoints))]
    matchedAveragePoints.sort(key=lambda coord: (coord[0] + coord[2]) - perp_slope * (coord[1] + coord[3]))
    lowerInterceptAveragePoints = [[matchedAveragePoints[i][0], matchedAveragePoints[i][1]] for i in
                                   range(len(matchedAveragePoints))]
    higherInterceptAveragePoints = [[matchedAveragePoints[i][2], matchedAveragePoints[i][3]] for i in
                                    range(len(matchedAveragePoints))]

    return (lowerInterceptAveragePoints, higherInterceptAveragePoints)


def getIdealPointGroup(points):
    pointGroups = []
    subgroup = [points[0]]

    for i in range(len(points) - 1):
        prevPoint = points[i]
        currentPoint = points[i + 1]

        if (abs(int(prevPoint[0]) - int(currentPoint[0])) <= 1) and (
                abs(int(prevPoint[1]) - int(currentPoint[1])) <= 1):
            subgroup.append(currentPoint)
        else:
            pointGroups.append(subgroup[:])
            subgroup = [currentPoint]

    pointGroups.append(subgroup)

    mainPointGroup = []
    maxPointGroupSize = 0

    for group in pointGroups:
        if len(group) > maxPointGroupSize:
            maxPointGroup = group
            maxPointGroupSize = len(group)

    return maxPointGroup


# Slope between points
def getSlope(point1, point2):
    if ((point1[0] == point2[0])):
        return -333
    return (point1[1] - point2[1]) / (point1[0] - point2[0])


def splitPoints(x1, y1, x2, y2, slope, points):
    p1Index = points.index([x1, y1])
    p2Index = points.index([x2, y2])

    lowerIndex = min(p1Index, p2Index)
    higherIndex = max(p1Index, p2Index)

    higherIntercept = points[lowerIndex:higherIndex]
    lowerIntercept = points[higherIndex:] + points[:lowerIndex]

    return (lowerIntercept, higherIntercept)


# Finds points for main contour line
def getTopAndBottomCoords(points):
    # Minimum and Maximum Y Coord
    maxY = max(points, key=lambda point: point[1])
    minY = min(points, key=lambda point: point[1])

    # MinY and MaxY With the limits
    minYWith5 = minY[1] + 5
    maxYWithout5 = maxY[1] - 15

    # Creating these arrays
    minYWith5Arr = []
    maxYWithout5Arr = []

    # Finding these points
    for point in points:
        if point[1] == minYWith5:
            minYWith5Arr.append(point)
        if point[1] == maxYWithout5:
            maxYWithout5Arr.append(point)

    # Average X Coordinates
    averageTopX = round((minYWith5Arr[0][0] + minYWith5Arr[-1][0]) / 2)
    averageBottomX = round((maxYWithout5Arr[0][0] + maxYWithout5Arr[-1][0]) / 2)
    slope = getSlope([averageTopX, minYWith5], [averageBottomX, maxYWithout5])

    averageTopX -= round((minYWith5Arr[-1][0] - minYWith5Arr[0][0]) / 1.5 / slope)
    averageBottomX += round((maxYWithout5Arr[-1][0] - maxYWithout5Arr[0][0]) / 3 / slope)

    # Creating these arrays
    averageTopXArr = []
    averageBottomXArr = []

    # Finding these points
    condition = True
    if slope > 0:
        while condition and averageTopX <= minYWith5Arr[-1][0] and averageBottomX >= maxYWithout5Arr[0][0]:
            for point in points:
                if point[0] == averageTopX:
                    averageTopXArr.append(point)
                if point[0] == averageBottomX:
                    averageBottomXArr.append(point)
            if len(averageTopXArr) > 0 and len(averageBottomXArr):
                condition = False
            if len(averageTopXArr) == 0:
                averageTopX += 1
            if len(averageBottomXArr) == 0:
                averageBottomXArr -= 1
    else:
        while condition and averageTopX >= minYWith5Arr[0][0] and averageBottomX <= maxYWithout5Arr[-1][0]:
            for point in points:
                if point[0] == averageTopX:
                    averageTopXArr.append(point)
                if point[0] == averageBottomX:
                    averageBottomXArr.append(point)
            if len(averageTopXArr) > 0 and len(averageBottomXArr):
                condition = False
            if len(averageTopXArr) == 0:
                averageTopX -= 1
            if len(averageBottomXArr) == 0:
                averageBottomXArr += 1

    # Sorting Arrs
    averageTopXArr.sort(key=lambda point: point[1])
    averageBottomXArr.sort(key=lambda point: point[1])
    averageBottomXArr.reverse()

    # Finding Min Top and Max Botpp,
    TopCoord = averageTopXArr[0]
    BottomCoord = averageBottomXArr[0]

    x1, y1 = TopCoord
    x2, y2 = BottomCoord

    return (x1, y1, x2, y2)


# Create the 20 equally spaced points
def getWeightedAveragePoints(x1, y1, x2, y2, number):
    weighted_avg = []

    for n in range(1, number + 1, 1):
        x_perpendicular = (((n * x1) + (number + 1 - n) * (x2)) / (number + 1))
        y_perpendicular = (((n * y1) + (number + 1 - n) * (y2)) / (number + 1))
        weighted_avg.append([x_perpendicular, y_perpendicular])

    for pair in weighted_avg:
        x, y = pair
        if x == int(x):
            pair[0] += 0.0001
        if y == int(y):
            pair[1] += 0.0001

    return weighted_avg


def calculateVolume(mask, number, sweeps=15, method="Method of Disks"):
    points = getIdealPointGroup(obtainContourPoints(mask))

    x1, y1, x2, y2 = getTopAndBottomCoords(points)
    if (x1 + y1) > (x2 + y2):
        x1, y1, x2, y2 = x2, y2, x1, y1

    mainLineSlope = getSlope([x1, y1], [x2, y2])
    baseAngle = math.atan(mainLineSlope)

    if baseAngle > 0:
        baseAngle -= math.pi
    lowerIntercept, higherIntercept = splitPoints(x1, y1, x2, y2, mainLineSlope, points)

    if (higherIntercept[0][0] + higherIntercept[0][1]) > (lowerIntercept[0][0] + lowerIntercept[0][1]):
        lowerIntercept, higherIntercept = higherIntercept, lowerIntercept

    volumes = {}
    x1s = {}
    y1s = {}
    x2s = {}
    y2s = {}
    degrees = {}

    # Volumes for all 0 to 5 cases
    for i in range(-sweeps, sweeps + 1, 1):
        x1, y1 = lowerIntercept[i]
        x2, y2 = higherIntercept[i]

        slope = getSlope([x1, y1], [x2, y2])
        angle = math.atan(slope)

        if angle > 0:
            angle -= math.pi

        degrees[i] = (baseAngle - angle) * 180 / math.pi

        p1Index = points.index([x1, y1])
        p2Index = points.index([x2, y2])

        lowerIndex = min(p1Index, p2Index)
        higherIndex = max(p1Index, p2Index)

        higherInterceptPoints = points[lowerIndex:higherIndex]
        lowerInterceptPoints = points[higherIndex:] + points[:lowerIndex]

        if (higherInterceptPoints[0][0] + higherInterceptPoints[0][1]) < (
                lowerInterceptPoints[0][0] + lowerInterceptPoints[0][1]):
            lowerInterceptPoints, higherInterceptPoints = higherInterceptPoints, lowerInterceptPoints

        weighted_avg = getWeightedAveragePoints(x1, y1, x2, y2, number)
        lowerInterceptAveragePoints, higherInterceptAveragePoints = findCorrespondingMaskPoints(weighted_avg,
                                                                                                lowerInterceptPoints,
                                                                                                higherInterceptPoints,
                                                                                                x1, y1, x2, y2, slope,
                                                                                                i)

        x1s[i] = [x1] + [point[0] for point in lowerInterceptAveragePoints]
        y1s[i] = [y1] + [point[1] for point in lowerInterceptAveragePoints]

        x2s[i] = [x2] + [point[0] for point in higherInterceptAveragePoints]
        y2s[i] = [y2] + [point[1] for point in higherInterceptAveragePoints]

        if method == "Method of Disks":
            volumes[i] = volumeMethodOfDisks(x1, y1, x2, y2, number, lowerInterceptAveragePoints,
                                             higherInterceptAveragePoints)
    return (volumes, x1s, y1s, x2s, y2s, degrees)


# Change to Method of Disks
def volumeMethodOfDisks(x1, y1, x2, y2, number, lowerInterceptAveragePoints, higherInterceptAveragePoints):
    # Long axis length and perp initialzation
    distance = getDistance([x1, y1], [x2, y2])
    parallelSeperationDistance = distance / (number + 1)

    lowerInterceptAveragePoints = np.asarray(lowerInterceptAveragePoints).swapaxes(1, 0)
    higherInterceptAveragePoints = np.asarray(higherInterceptAveragePoints).swapaxes(1, 0)
    # Simpson Volume Methods
    volume = 0

    for i in range(len(lowerInterceptAveragePoints)):
        diameter = getDistance(lowerInterceptAveragePoints[i], higherInterceptAveragePoints[i])
        radius = diameter / 2
        diskVolume = math.pi * radius ** 2 * parallelSeperationDistance
        volume += diskVolume

    return volume
