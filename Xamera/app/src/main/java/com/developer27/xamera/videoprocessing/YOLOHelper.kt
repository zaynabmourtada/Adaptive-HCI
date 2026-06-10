package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

object YOLOHelper {

    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        val detections = mutableListOf<DetectionResult>()
        for (i in 0 until numDetections) {
            val confidence = rawOutput[0][4][i]
            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(
                    xCenter = rawOutput[0][0][i],
                    yCenter = rawOutput[0][1][i],
                    width = rawOutput[0][2][i],
                    height = rawOutput[0][3][i],
                    confidence = confidence
                ))
            }
        }
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }

        val detectionBoxes = detections.map { it to detectionToBox(it) }.sortedByDescending { it.first.confidence }.toMutableList()
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { computeIoU(current.second, it.second) > Settings.Inference.iouThreshold }
        }

        return nmsDetections.maxByOrNull { it.confidence }?.also { d ->
            Log.d("YOLOTest", "Best detection: confidence=${"%.8f".format(d.confidence)}, x=${d.xCenter}, y=${d.yCenter}")
        }
    }

    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2,
        y1 = d.yCenter - d.height / 2,
        x2 = d.xCenter + d.width / 2,
        y2 = d.yCenter + d.height / 2,
        confidence = d.confidence
    )

    private fun computeIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
        return if (union > 0f) intersection / union else 0f
    }

    fun rescaleDetection(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        letterbox: LetterboxResult,
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {
        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val xCenterOriginal = (xCenterLetterboxed - letterbox.padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - letterbox.padTop) / scale
        val boxWidthOriginal = detection.width * modelInputWidth / scale
        val boxHeightOriginal = detection.height * modelInputHeight / scale

        val box = BoundingBox(
            x1 = (xCenterOriginal - boxWidthOriginal / 2).toFloat(),
            y1 = (yCenterOriginal - boxHeightOriginal / 2).toFloat(),
            x2 = (xCenterOriginal + boxWidthOriginal / 2).toFloat(),
            y2 = (yCenterOriginal + boxHeightOriginal / 2).toFloat(),
            confidence = detection.confidence
        )
        Log.d("YOLOTest", "Adjusted box: x1=${"%.8f".format(box.x1)}, y1=${"%.8f".format(box.y1)}, x2=${"%.8f".format(box.x2)}, y2=${"%.8f".format(box.y2)}")
        return box to Point(xCenterOriginal, yCenterOriginal)
    }

    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)

        val label = "User_1 (${"%.2f".format(box.confidence * 100)}%)"
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
        val textY = (box.y1 - 5).toInt().coerceAtLeast(10)
        Imgproc.rectangle(
            mat,
            Point(textX.toDouble(), textY.toDouble() + baseline[0]),
            Point(textX + textSize.width, textY - textSize.height),
            Settings.BoundingBox.boxColor,
            Imgproc.FILLED
        )
        Imgproc.putText(mat, label, Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255.0, 255.0, 255.0), thickness)
    }

    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): LetterboxResult {
        val srcMat = Mat()
        Utils.bitmapToMat(srcBitmap, srcMat)
        val srcWidth = srcMat.cols().toDouble()
        val srcHeight = srcMat.rows().toDouble()

        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()

        val resized = Mat()
        Imgproc.resize(srcMat, resized, Size(newWidth.toDouble(), newHeight.toDouble()))
        srcMat.release()

        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        val top = padHeight / 2
        val bottom = padHeight - top
        val left = padWidth / 2
        val right = padWidth - left

        val letterboxed = Mat()
        Core.copyMakeBorder(resized, letterboxed, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)
        resized.release()

        val outputBitmap = Bitmap.createBitmap(letterboxed.cols(), letterboxed.rows(), srcBitmap.config ?: Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(letterboxed, outputBitmap)
        letterboxed.release()

        return LetterboxResult(outputBitmap, left, top)
    }
}
