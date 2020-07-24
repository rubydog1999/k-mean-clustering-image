import numpy as np
import cv2
from random import randrange
from math import sqrt
import pandas

image = cv2.imread("C:/Users/Trung Dung/PycharmProjects/Clustering/img/Test1.jpg", 1)
Blue = image[:, :, 0]
Red = image[:, :, 1]
Green = image[:, :, 2]
# Table for pixel RGB

blue_1D = Blue.reshape(-1, 1)
red_1D = Red.reshape(-1, 1)
green_1D = Green.reshape(-1, 1)
table_RGB = np.concatenate((blue_1D, red_1D, green_1D), axis=1)
row_lable_RGB = (blue_1D.shape[0])
column_names_XY = ['pixel B', 'pixel R', 'pixel G']
table_pixelRGB = pandas.DataFrame(table_RGB, columns=column_names_XY, index=range(row_lable_RGB))
print("table for Pixel RGB")
print(table_pixelRGB)

# table for pixel X pixel Y
pixel_X = image.shape[0]
pixel_Y = image.shape[1]
Pixel_X = []
Pixel_Y = []
for a in range(pixel_X):
    Pixel_X.append(a)
for b in range(pixel_Y):
    Pixel_Y.append(b)

if Pixel_Y < Pixel_X:
    rows_label = pixel_X
    for c in range(pixel_Y, pixel_X):
        Pixel_Y.append('NaN')
elif Pixel_X < Pixel_Y:
    rows_label = pixel_Y
    for c in range(pixel_X, pixel_Y):
        Pixel_X.append('NaN')
else:
    rows_label = pixel_Y = pixel_X

pixel_y = np.array(Pixel_Y)
pixel_x = np.array(Pixel_X)
pixel_x_1D = pixel_x.reshape(-1, 1)
pixel_y_1D = pixel_y.reshape(-1, 1)
table_pixelxy = np.concatenate((pixel_y_1D, pixel_x_1D), axis=1)
column_names = ['pixel X', 'pixel Y']
table_pixelXY = pandas.DataFrame(table_pixelxy, columns=column_names, index=range(rows_label))
print(table_pixelXY)


# kmean implement
def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def point_rgb(point, img):
    return img[point[0], point[1]]


def k_means(img, k, max_iter):
    Array = []
    centers = []
    clusters = {}
    for i in range(k):
        center = (randrange(len(img)), randrange(len(img[0])))
        if center not in centers:
            centers.append(center)
            clusters[center] = []
    for i in range(max_iter):
        for x in range(len(img)):
            for y in range(len(img[0])):
                rgb1 = img[x][y]
                min_distance = distance(rgb1, point_rgb(centers[0], img=img))
                best_cluster = centers[0]
                for center in centers:
                    rgb2 = point_rgb(center, img)
                    dist = distance(rgb1, rgb2)
                    if dist < min_distance:
                        min_distance = dist
                        best_cluster = center
                clusters[best_cluster].append((rgb1, [x, y]))

        avg = []
        for c in clusters:
            if len(clusters[c]) == 0:
                break
            sum_red = 0
            sum_blue = 0
            sum_green = 0
            for point in clusters[c]:
                sum_red += point[0][0]
                sum_green += point[0][1]
                sum_blue += point[0][2]
            red = sum_red / len(clusters[c])
            blue = sum_blue / len(clusters[c])
            green = sum_green / len(clusters[c])
            avg.append([red, green, blue])

        centers = []
        clusters = {}
        for a in avg:
            min_distance = distance(a, img[0][0])
            best_clust = img[0][0]
            for row in range(len(img)):
                for column in range(len(img[0])):
                    new_dist = distance(a, img[row][column])
                    if new_dist < min_distance:
                        min_distance = new_dist
                        best_clust = (row, column)
            centers.append(best_clust)
            clusters[best_clust] = []

    for x in range(len(img)):
        for y in range(len(img[0])):
            rgb1 = img[x][y]
            min_distance = distance(rgb1, point_rgb(centers[0], img=img))
            best_cluster = centers[0]
            for center in centers:
                rgb2 = point_rgb(center, img)
                dist = distance(rgb1, rgb2)
                if dist < min_distance:
                    min_distance = dist
                    best_cluster = center
            clusters[best_cluster].append((best_cluster,rgb1, [x, y], min_distance))
    print("Average pixel RGB for " + " " + str(k) + " " + "cluster")
    print(avg)
    return clusters



cl = k_means(img=image, k=2, max_iter=10)
# data = list(cl.items())
# an_array = np.array(data)
table_element_2 =[]
for j in cl:
    for z in cl[j]:
        table_element_2.append(z)
row = len(table_element_2)
column_names = ["Cluster", "Pixel RGB", "pixel X Y", "distance to center mass"]
table_cluster = pandas.DataFrame(table_element_2, columns=column_names, index=range(row))

print(table_cluster)
# row_names = []
# table_clusters = np.concatenate((table_element1,table_element2),axis=1)
# for o in range(2):
#     row_names.append('Cluster')
# print(column_names)
column_names=["Pixel XY center mass","Pixel RGB, pixel X Y and distance to center mass"]
# table_cluster = pandas.DataFrame(table_clusters, columns=column_names, index=row_names)
# print(table_cluster)
# print(table_clusters)


# table_cluster = np.concatenate((Cluster,Element_cluster), axis=1)
# row_lable_RGB = (blue_1D.shape[0])
# column_names_XY = ['pixel B', 'pixel R', 'pixel G']
# table_pixelRGB = pandas.DataFrame(table_RGB, columns=column_names_XY, index=range(row_lable_RGB))
# print("table for Pixel RGB")
# print(table_pixelRGB)

