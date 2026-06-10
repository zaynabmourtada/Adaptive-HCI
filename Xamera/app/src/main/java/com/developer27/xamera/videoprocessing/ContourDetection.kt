package com.developer27.xamera.videoprocessing

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

object ContourDetection {

    fun processContourDetection(mat: Mat): List<Rect> {
        val contours = findContours(mat)
        val rois = contours.map { Imgproc.boundingRect(it) }.toMutableList()
        return mergeCloseRois(rois)
    }

    private fun findContours(mat: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours.filter { Imgproc.contourArea(it) > 5000 }
    }

    private fun mergeCloseRois(rois: MutableList<Rect>): MutableList<Rect> {
        val merged = rois.toMutableList()
        var didMerge = true
        while (didMerge) {
            didMerge = false
            outer@ for (i in 0 until merged.size) {
                for (j in i + 1 until merged.size) {
                    if (areClose(merged[i], merged[j])) {
                        val union = union(merged[i], merged[j])
                        merged.removeAt(j)
                        merged.removeAt(i)
                        merged.add(union)
                        didMerge = true
                        break@outer
                    }
                }
            }
        }
        return merged
    }

    private fun areClose(r1: Rect, r2: Rect): Boolean {
        val e1 = Rect(r1.x - 10, r1.y - 10, r1.width + 20, r1.height + 20)
        val e2 = Rect(r2.x - 10, r2.y - 10, r2.width + 20, r2.height + 20)
        return e1.x < e2.x + e2.width &&
                e1.x + e1.width > e2.x &&
                e1.y < e2.y + e2.height &&
                e1.y + e1.height > e2.y
    }

    private fun union(r1: Rect, r2: Rect): Rect {
        val x = min(r1.x, r2.x)
        val y = min(r1.y, r2.y)
        val x2 = max(r1.x + r1.width, r2.x + r2.width)
        val y2 = max(r1.y + r1.height, r2.y + r2.height)
        return Rect(x, y, x2 - x, y2 - y)
    }
}
