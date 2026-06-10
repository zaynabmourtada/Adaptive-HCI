package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class VideoProcessor {

    companion object {
        init {
            try {
                System.loadLibrary("opencv_java4")
            } catch (e: UnsatisfiedLinkError) {
                Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
            }
        }
    }

    @Volatile private var tfliteInterpreter: Interpreter? = null

    private val listLock = Any()
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    private val kalmanHelper = KalmanHelper()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    fun setInterpreter(model: Interpreter) {
        tfliteInterpreter = model
        Log.d("VideoProcessor", "TFLite model set successfully.")
    }

    fun reset() {
        synchronized(listLock) {
            rawDataList.clear()
            smoothDataList.clear()
        }
    }

    fun close() {
        scope.cancel()
        kalmanHelper.close()
    }

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        scope.launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameContour(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameYolo(bitmap)
                }
            } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    private fun processFrameContour(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        val sourceMat = Mat()
        return try {
            Utils.bitmapToMat(bitmap, sourceMat)

            val preprocessedMat = Preprocessing.preprocessFrame(bitmap)
            val rois = try {
                ContourDetection.processContourDetection(preprocessedMat)
            } finally {
                preprocessedMat.release()
            }

            if (Settings.BoundingBox.enableBoundingBox) {
                for (roi in rois) {
                    Imgproc.rectangle(sourceMat, roi.tl(), roi.br(), Scalar(255.0, 0.0, 0.0), 3)
                }
            }
            val center = rois.maxByOrNull { it.width * it.height }
                ?.let { Point(it.x + it.width / 2.0, it.y + it.height / 2.0) }

            synchronized(listLock) {
                TraceRenderer.drawTrace(center, sourceMat, kalmanHelper, rawDataList, smoothDataList)
            }

            val outBitmap = Bitmap.createBitmap(sourceMat.cols(), sourceMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(sourceMat, outBitmap)
            outBitmap to outBitmap
        } catch (e: Exception) {
            Log.d("VideoProcessor", "Error in contour processing: ${e.message}", e)
            null
        } finally {
            sourceMat.release()
        }
    }

    private suspend fun processFrameYolo(bitmap: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.Default) {
            val dims = getModelDimensions()
            val letterbox = YOLOHelper.createLetterboxedBitmap(bitmap, dims.inputWidth, dims.inputHeight)
            val frameMat = Mat()
            Utils.bitmapToMat(bitmap, frameMat)

            try {
                if (Settings.DetectionMode.enableYoloInference && tfliteInterpreter != null) {
                    val out = Array(dims.outputShape[0]) { Array(dims.outputShape[1]) { FloatArray(dims.outputShape[2]) } }
                    TensorImage(DataType.FLOAT32).apply { load(letterbox.bitmap) }
                        .also { tfliteInterpreter?.run(it.buffer, out) }

                    YOLOHelper.parseTFLite(out)?.let { detection ->
                        val (box, center) = YOLOHelper.rescaleDetection(
                            detection, bitmap.width, bitmap.height, letterbox, dims.inputWidth, dims.inputHeight
                        )
                        if (Settings.BoundingBox.enableBoundingBox) YOLOHelper.drawBoundingBoxes(frameMat, box)
                        synchronized(listLock) {
                            TraceRenderer.drawTrace(center, frameMat, kalmanHelper, rawDataList, smoothDataList)
                        }
                    }
                }

                val yoloBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(frameMat, yoloBitmap)
                yoloBitmap to letterbox.bitmap
            } finally {
                frameMat.release()
            }
        }

    fun getModelDimensions(): ModelDimensions {
        val inShape = tfliteInterpreter?.getInputTensor(0)?.shape()
        val outShape = tfliteInterpreter?.getOutputTensor(0)?.shape()?.toList() ?: listOf(1, 5, 3549)
        return ModelDimensions(
            inputWidth = inShape?.getOrNull(2) ?: 416,
            inputHeight = inShape?.getOrNull(1) ?: 416,
            outputShape = outShape
        )
    }

    fun exportTraceForInference(): Bitmap {
        val points = synchronized(listLock) { smoothDataList.toList() }
        if (points.isEmpty()) {
            return Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }
        val bounds = computeTraceBounds(points)
            ?: return Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }

        val padding = 30.0
        val optimalWidth = max((bounds.maxX - bounds.minX + 2 * padding).toInt(), 1)
        val optimalHeight = max((bounds.maxY - bounds.minY + 2 * padding).toInt(), 1)
        val squareSize = max(optimalWidth, optimalHeight)

        val mat = Mat(squareSize, squareSize, CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))
        val xOffset = (squareSize - optimalWidth) / 2.0
        val yOffset = (squareSize - optimalHeight) / 2.0
        val adjusted = points.map {
            Point(it.x - bounds.minX + padding + xOffset, it.y - bounds.minY + padding + yOffset)
        }

        TraceRenderer.drawSplineCurve(adjusted, mat, Scalar(0.0, 0.0, 0.0), 40)

        val outputBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outputBitmap)
        mat.release()
        return Bitmap.createScaledBitmap(outputBitmap, 28, 28, true)
    }

    fun exportTracesAsBitmap(): Bitmap {
        val (rawPoints, smoothPoints) = synchronized(listLock) {
            rawDataList.toList() to smoothDataList.toList()
        }
        val allPoints = rawPoints + smoothPoints
        if (allPoints.isEmpty()) {
            return Bitmap.createBitmap(500, 500, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }

        val bounds = computeTraceBounds(allPoints)!!
        val padding = 50.0
        val scaleFactor = 2.0
        val baseWidth = (bounds.maxX - bounds.minX + 2 * padding).coerceAtLeast(1.0)
        val baseHeight = (bounds.maxY - bounds.minY + 2 * padding).coerceAtLeast(1.0)
        val width = (baseWidth * scaleFactor).toInt()
        val height = (baseHeight * scaleFactor).toInt()

        val mat = Mat(height, width, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))

        fun toScaled(pt: Point) = Point(
            (pt.x - bounds.minX + padding) * scaleFactor,
            (pt.y - bounds.minY + padding) * scaleFactor
        )

        if (rawPoints.size > 1) {
            for (i in 1 until rawPoints.size) {
                Imgproc.line(mat, toScaled(rawPoints[i - 1]), toScaled(rawPoints[i]),
                    Settings.Trace.originalLineColor, Settings.Trace.lineThickness)
            }
        }

        if (smoothPoints.size >= 3) {
            val (splineX, splineY) = TraceRenderer.applySplineInterpolation(smoothPoints)
            var prev: Point? = null
            var t = 0.0
            while (t <= smoothPoints.size - 1.0) {
                val cur = Point(
                    (splineX.value(t) - bounds.minX + padding) * scaleFactor,
                    (splineY.value(t) - bounds.minY + padding) * scaleFactor
                )
                prev?.let { Imgproc.line(mat, it, cur, Settings.Trace.splineLineColor, Settings.Trace.lineThickness) }
                prev = cur
                t += Settings.Trace.splineStep
            }
        }

        val rawLength = measurePathLength(rawPoints)
        val midLineX = ((bounds.minX + bounds.maxX) / 2.0 - bounds.minX + padding) * scaleFactor
        val startY = max(0.0, (baseHeight - rawLength) / 2.0) * scaleFactor
        Imgproc.line(mat, Point(midLineX, startY), Point(midLineX, startY + rawLength * scaleFactor),
            Scalar(0.0, 0.0, 0.0), 2)

        val outBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outBitmap)
        mat.release()
        return outBitmap
    }

    fun getTrackingCoordinatesString(): String {
        return synchronized(listLock) {
            smoothDataList.joinToString(separator = ";") { "${it.x},${it.y},0.0" }
        }
    }

    private fun computeTraceBounds(points: List<Point>): TraceBounds? {
        if (points.isEmpty()) return null
        var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE
        for (pt in points) {
            minX = min(minX, pt.x); minY = min(minY, pt.y)
            maxX = max(maxX, pt.x); maxY = max(maxY, pt.y)
        }
        return TraceBounds(minX, minY, maxX, maxY)
    }

    private fun measurePathLength(points: List<Point>): Double {
        var total = 0.0
        for (i in 1 until points.size) {
            val dx = points[i].x - points[i - 1].x
            val dy = points[i].y - points[i - 1].y
            total += sqrt(dx * dx + dy * dy)
        }
        return total
    }
}
