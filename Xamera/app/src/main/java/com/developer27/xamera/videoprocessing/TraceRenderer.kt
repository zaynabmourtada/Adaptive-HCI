package com.developer27.xamera.videoprocessing

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.LinkedList

object TraceRenderer {

    fun drawTrace(
        center: Point?,
        mat: Mat,
        kalmanHelper: KalmanHelper,
        rawList: LinkedList<Point>,
        smoothList: LinkedList<Point>
    ) {
        if (center != null) {
            rawList.add(center)
            val (fx, fy) = kalmanHelper.filter(center)
            smoothList.add(Point(fx, fy))
            if (rawList.size > Settings.Trace.lineLimit) rawList.pollFirst()
            if (smoothList.size > Settings.Trace.lineLimit) smoothList.pollFirst()
        }
        if (Settings.Trace.enableRawTrace) drawRawTrace(rawList, mat)
        if (Settings.Trace.enableSplineTrace) drawSplineCurve(
            smoothList, mat,
            Settings.Trace.splineLineColor,
            Settings.Trace.lineThickness
        )
    }

    fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(image, data[i - 1], data[i], Settings.Trace.originalLineColor, Settings.Trace.lineThickness)
        }
    }

    fun drawSplineCurve(data: List<Point>, image: Mat, color: Scalar, thickness: Int) {
        if (data.size < 3) return
        val (splineX, splineY) = applySplineInterpolation(data)
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let { Imgproc.line(image, it, currentPoint, color, thickness) }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }

    fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction> {
        val interpolator = SplineInterpolator()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()
        val splineX = interpolator.interpolate(tData, data.map { it.x }.toDoubleArray())
        val splineY = interpolator.interpolate(tData, data.map { it.y }.toDoubleArray())
        return splineX to splineY
    }
}
