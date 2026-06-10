package com.developer27.xamera.videoprocessing

import org.opencv.core.Scalar

object Settings {
    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.YOLO
        var enableYoloInference = false
    }
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
    }
    object Trace {
        var enableRawTrace = false
        var enableSplineTrace = true
        var lineLimit = 75
        var splineStep = 0.01
        var originalLineColor = Scalar(0.0, 39.0, 76.0)
        var splineLineColor = Scalar(255.0, 203.0, 5.0)
        var lineThickness = 4
    }
    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(0.0, 39.0, 76.0)
        var boxThickness = 2
    }
    object Brightness {
        var factor = 2.0
        var threshold = 10.0
    }
    object ExportData {
        var frameIMG = true
        var videoDATA = true
    }
}
