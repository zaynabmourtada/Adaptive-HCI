package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap

data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float
)

data class ModelDimensions(
    val inputWidth: Int,
    val inputHeight: Int,
    val outputShape: List<Int>
)

data class LetterboxResult(
    val bitmap: Bitmap,
    val padLeft: Int,
    val padTop: Int
)

data class TraceBounds(
    val minX: Double,
    val minY: Double,
    val maxX: Double,
    val maxY: Double
)
