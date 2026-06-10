package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object Preprocessing {
    fun preprocessFrame(src: Bitmap): Mat {
        val sourceMat = Mat()
        Utils.bitmapToMat(src, sourceMat)

        val grayMat = Mat()
        Imgproc.cvtColor(sourceMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        sourceMat.release()

        val brightenedMat = Mat()
        Core.multiply(grayMat, Scalar(Settings.Brightness.factor), brightenedMat)
        grayMat.release()

        val thresholdMat = Mat()
        Imgproc.threshold(brightenedMat, thresholdMat, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO)
        brightenedMat.release()

        val blurredMat = Mat()
        Imgproc.GaussianBlur(thresholdMat, blurredMat, Size(5.0, 5.0), 0.0)
        thresholdMat.release()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val closedMat = Mat()
        Imgproc.morphologyEx(blurredMat, closedMat, Imgproc.MORPH_CLOSE, kernel)
        blurredMat.release()

        return closedMat
    }
}
