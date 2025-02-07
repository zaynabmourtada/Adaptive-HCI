package com.developer27.xamera.videoprocessing

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.opengl.GLUtils
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.LinkedList

class ProcessedVideoRecorder(
    private val width: Int,
    private val height: Int,
    private val outputFilePath: String
) {
    private val TAG = "ProcessedVideoRecorder"
    private val MIME_TYPE = "video/avc"
    private val FRAME_RATE = 30
    private val IFRAME_INTERVAL = 1
    private val BIT_RATE = 6000000

    private var mediaCodec: MediaCodec? = null
    private var mediaMuxer: MediaMuxer? = null
    private var trackIndex = -1
    private var muxerStarted = false
    private var presentationTimeUs: Long = 0

    // Lists for processing/tracking (same as in your VideoProcessor).
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    fun start() {
        try {
            val format = MediaFormat.createVideoFormat(MIME_TYPE, width, height)
            format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
            )
            format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE)
            format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE)
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, IFRAME_INTERVAL)

            mediaCodec = MediaCodec.createEncoderByType(MIME_TYPE)
            mediaCodec?.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            mediaCodec?.start()

            mediaMuxer = MediaMuxer(outputFilePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxerStarted = false
            presentationTimeUs = 0
            Log.d(TAG, "Started recording to: $outputFilePath")
        } catch (e: Exception) {
            Log.e(TAG, "Error during start: ${e.message}", e)
        }
    }

    /**
     * Convert and encode each frame.
     */
    fun recordFrame(bitmap: Bitmap) {
        // Scale the bitmap if it doesn't match the configured size.
        val scaledBitmap = if (bitmap.width != width || bitmap.height != height)
            Bitmap.createScaledBitmap(bitmap, width, height, false)
        else
            bitmap

        // Process the frame (apply rolling shutter, detection, etc.)
        // Then use OpenGL to overlay the line.
        val processedBitmap = processFrameInternalCONTOUR(scaledBitmap)
        if (processedBitmap == null) {
            Log.e(TAG, "Processed frame is null; skipping frame.")
            return
        }

        // Convert the processed Bitmap (with the OpenGL-drawn overlay) to NV12 (YUV).
        val input = convertBitmapToYUV(processedBitmap) ?: run {
            Log.e(TAG, "Conversion to YUV failed!")
            return
        }

        mediaCodec?.let { codec ->
            val inputBufferIndex = codec.dequeueInputBuffer(10_000)
            if (inputBufferIndex >= 0) {
                val inputBuffer: ByteBuffer? = codec.getInputBuffer(inputBufferIndex)
                inputBuffer?.clear()
                inputBuffer?.put(input)
                codec.queueInputBuffer(inputBufferIndex, 0, input.size, presentationTimeUs, 0)
                presentationTimeUs += 1_000_000L / FRAME_RATE
            }
            drainEncoder(false)
        }
    }

    /**
     * 1) Convert the input Bitmap to a Mat.
     * 2) Optionally run detection to update the points lists.
     * 3) Apply a rolling shutter effect.
     * 4) Instead of drawing with OpenCV, convert the processed Mat to a Bitmap and
     *    then use OpenGL to render the overlay “line” (spline or fallback) on top.
     * 5) Return the final Bitmap.
     */
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Bitmap? {
        val originalMat = Mat()
        try {
            // 1) Convert Bitmap to Mat.
            Utils.bitmapToMat(bitmap, originalMat)
            if (originalMat.channels() == 1) {
                Imgproc.cvtColor(originalMat, originalMat, Imgproc.COLOR_GRAY2BGR)
            }

            // 2) [Optional] Clone for detection.
            val detectionMat = originalMat.clone()
            val preprocessedMat = Preprocessing.preprocessFrame(detectionMat)
            val (center, processedMat) = ContourDetection.processContourDetection(preprocessedMat)
            // Prepare data for line drawing.
            if (center != null) {
                rawDataList.add(center)

                // Apply Kalman filter to smooth tracking
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))

                // Maintain trace history within limits
                if (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
                if (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()

                // Draw raw trace and smoothed trace directly on preprocessedMat
                //TraceRenderer.drawRawTrace(rawDataList, preprocessedMat)
                TraceRenderer.drawSplineCurve(smoothDataList, preprocessedMat)
            }

            // 3) Apply rolling shutter effect.
            val rolledMat = applyRollingShutterEffect(originalMat)

            // (Optional) You could do additional processing here.
            // Convert the processed Mat to a Bitmap.
            val baseBitmap = Bitmap.createBitmap(rolledMat.cols(), rolledMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(rolledMat, baseBitmap)

            // 4) Use OpenGL to overlay the line on top of the base image.
            val hasContour = (center != null)
            val outputBitmap = renderWithOpenGLOverlay(baseBitmap, smoothDataList, hasContour)
            return outputBitmap

        } catch (e: Exception) {
            Log.e(TAG, "Error in processFrameInternalCONTOUR: ${e.message}", e)
            return null
        } finally {
            originalMat.release()
        }
    }

    /**
     * Applies a basic rolling shutter effect by shifting each row horizontally.
     */
    private fun applyRollingShutterEffect(src: Mat): Mat {
        val rows = src.rows()
        val cols = src.cols()
        val channels = src.channels()  // e.g., 3 for a color image.

        val rolledMat = Mat(rows, cols, src.type())
        val SHIFT_SCALE = 0.2
        val rowBuffer = ByteArray(cols * channels)

        for (row in 0 until rows) {
            src.get(row, 0, rowBuffer)
            val shift = (row * SHIFT_SCALE).toInt()
            for (col in 0 until cols) {
                val readCol = col + shift
                val clampCol = when {
                    readCol < 0 -> 0
                    readCol >= cols -> cols - 1
                    else -> readCol
                }
                val readIdx = clampCol * channels
                rolledMat.put(row, col, rowBuffer, readIdx, channels)
            }
        }
        return rolledMat
    }

    /**
     * Uses an offscreen OpenGL ES 2.0 context to overlay a line (either a spline
     * drawn from the provided points or a fallback horizontal line) on top of the baseBitmap.
     *
     * The steps are:
     *   1. Set up an EGL pbuffer surface.
     *   2. Create a texture from the baseBitmap.
     *   3. Render a full-screen textured quad.
     *   4. Render the line overlay (using the provided points or fallback).
     *   5. Read the rendered pixels into a Bitmap.
     */
    private fun renderWithOpenGLOverlay(baseBitmap: Bitmap, points: LinkedList<Point>, hasContour: Boolean): Bitmap? {
        // 1. Set up EGL.
        val eglDisplay: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "Unable to get EGL display")
            return null
        }
        val version = IntArray(2)
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
        val eglConfig = configs[0]

        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH, baseBitmap.width,
            EGL14.EGL_HEIGHT, baseBitmap.height,
            EGL14.EGL_NONE
        )
        val eglSurface: EGLSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs, 0)

        val contextAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )
        val eglContext: EGLContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)

        // 2. Set up viewport.
        GLES20.glViewport(0, 0, baseBitmap.width, baseBitmap.height)

        // 3. Create a texture from the baseBitmap.
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        val baseTextureId = textures[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, baseTextureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, baseBitmap, 0)

        // 4. Render a full-screen textured quad.
        val quadProgram = createProgram(VERTEX_SHADER, FRAGMENT_SHADER_TEXTURE)
        GLES20.glUseProgram(quadProgram)
        val quadVertices = floatArrayOf(
            // Positions         // TexCoords
            -1f,  1f, 0f,       0f, 0f,  // top-left
            -1f, -1f, 0f,       0f, 1f,  // bottom-left
            1f,  1f, 0f,       1f, 0f,  // top-right
            1f, -1f, 0f,       1f, 1f   // bottom-right
        )
        val vertexBuffer = ByteBuffer.allocateDirect(quadVertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        vertexBuffer.put(quadVertices)
        vertexBuffer.position(0)
        val posHandle = GLES20.glGetAttribLocation(quadProgram, "a_Position")
        val texHandle = GLES20.glGetAttribLocation(quadProgram, "a_TexCoord")
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, 5 * 4, vertexBuffer)
        vertexBuffer.position(3)
        GLES20.glEnableVertexAttribArray(texHandle)
        GLES20.glVertexAttribPointer(texHandle, 2, GLES20.GL_FLOAT, false, 5 * 4, vertexBuffer)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        // 5. Draw the line overlay.
        val lineProgram = createProgram(VERTEX_SHADER, FRAGMENT_SHADER_COLOR)
        GLES20.glUseProgram(lineProgram)
        // Set the uniform color.
        // For example, if you want to use blue (from Settings.Trace.splineLineColor),
        // convert its components (assumed in BGR) to normalized RGBA.
        GLES20.glUniform4f(GLES20.glGetUniformLocation(lineProgram, "u_Color"), 0f, 0f, 1f, 1f)

        // Build vertex data: if points exist, use them; else draw fallback horizontal line.
        val lineVertices: FloatArray = if (hasContour && points.size > 1) {
            val vertexList = mutableListOf<Float>()
            for (p in points) {
                // Convert pixel coordinates (with origin at top-left) to normalized device coordinates.
                val ndcX = (p.x.toFloat() / baseBitmap.width) * 2f - 1f
                val ndcY = 1f - (p.y.toFloat() / baseBitmap.height) * 2f
                vertexList.add(ndcX)
                vertexList.add(ndcY)
                vertexList.add(0f)
            }
            vertexList.toFloatArray()
        } else {
            // Fallback: horizontal line across the center.
            floatArrayOf(-1f, 0f, 0f, 1f, 0f, 0f)
        }
        val lineVertexBuffer = ByteBuffer.allocateDirect(lineVertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        lineVertexBuffer.put(lineVertices)
        lineVertexBuffer.position(0)
        val linePosHandle = GLES20.glGetAttribLocation(lineProgram, "a_Position")
        GLES20.glEnableVertexAttribArray(linePosHandle)
        GLES20.glVertexAttribPointer(linePosHandle, 3, GLES20.GL_FLOAT, false, 3 * 4, lineVertexBuffer)
        GLES20.glLineWidth(Settings.Trace.lineThickness.toFloat())
        val vertexCount = if (hasContour && points.size > 1) points.size else 2
        GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, vertexCount)

        // 6. Finish rendering and read pixels.
        GLES20.glFinish()
        val buffer = ByteBuffer.allocateDirect(baseBitmap.width * baseBitmap.height * 4)
        GLES20.glReadPixels(0, 0, baseBitmap.width, baseBitmap.height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer)
        val resultBitmap = Bitmap.createBitmap(baseBitmap.width, baseBitmap.height, Bitmap.Config.ARGB_8888)
        resultBitmap.copyPixelsFromBuffer(buffer)

        // 7. Clean up.
        GLES20.glDeleteTextures(1, textures, 0)
        GLES20.glDeleteProgram(quadProgram)
        GLES20.glDeleteProgram(lineProgram)
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(eglDisplay, eglSurface)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)

        return resultBitmap
    }

    /**
     * Compiles and links the provided vertex and fragment shader source into an OpenGL program.
     */
    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        val program = GLES20.glCreateProgram()
        if (program == 0) {
            Log.e(TAG, "Could not create program")
            return 0
        }
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        val linkStatus = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] != GLES20.GL_TRUE) {
            Log.e(TAG, "Could not link program: " + GLES20.glGetProgramInfoLog(program))
            GLES20.glDeleteProgram(program)
            return 0
        }
        return program
    }

    /**
     * Loads and compiles a shader of the given type.
     */
    private fun loadShader(type: Int, shaderSource: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, shaderSource)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            Log.e(TAG, "Could not compile shader $type: " + GLES20.glGetShaderInfoLog(shader))
            GLES20.glDeleteShader(shader)
            return 0
        }
        return shader
    }

    // Simple shader source strings.
    private val VERTEX_SHADER = """
        attribute vec4 a_Position;
        attribute vec2 a_TexCoord;
        varying vec2 v_TexCoord;
        void main() {
            gl_Position = a_Position;
            v_TexCoord = a_TexCoord;
        }
    """.trimIndent()

    private val FRAGMENT_SHADER_TEXTURE = """
        precision mediump float;
        uniform sampler2D u_Texture;
        varying vec2 v_TexCoord;
        void main() {
            gl_FragColor = texture2D(u_Texture, v_TexCoord);
        }
    """.trimIndent()

    private val FRAGMENT_SHADER_COLOR = """
        precision mediump float;
        uniform vec4 u_Color;
        void main() {
            gl_FragColor = u_Color;
        }
    """.trimIndent()

    /**
     * Drains the encoder’s output buffers and writes them to the muxer.
     */
    private fun drainEncoder(endOfStream: Boolean) {
        if (endOfStream) {
            mediaCodec?.let { codec ->
                val inputBufferIndex = codec.dequeueInputBuffer(10_000)
                if (inputBufferIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputBufferIndex)
                    inputBuffer?.clear()
                    codec.queueInputBuffer(
                        inputBufferIndex,
                        0,
                        0,
                        presentationTimeUs,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    )
                }
            }
        }
        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferIndex = mediaCodec?.dequeueOutputBuffer(bufferInfo, 10_000) ?: break
            when {
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) break
                }
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    mediaCodec?.outputFormat?.let { newFormat ->
                        trackIndex = mediaMuxer?.addTrack(newFormat) ?: -1
                        mediaMuxer?.start()
                        muxerStarted = true
                    }
                }
                outputBufferIndex >= 0 -> {
                    val encodedData = mediaCodec?.getOutputBuffer(outputBufferIndex)
                    if (encodedData == null) {
                        Log.e(TAG, "Encoder output buffer $outputBufferIndex was null")
                        continue
                    }
                    if (bufferInfo.size != 0) {
                        if (!muxerStarted) throw RuntimeException("Muxer hasn't started")
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        mediaMuxer?.writeSampleData(trackIndex, encodedData, bufferInfo)
                    }
                    mediaCodec?.releaseOutputBuffer(outputBufferIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                }
            }
        }
    }

    fun stop() {
        try {
            drainEncoder(true)
            mediaCodec?.stop()
            mediaCodec?.release()
            mediaCodec = null

            mediaMuxer?.stop()
            mediaMuxer?.release()
            mediaMuxer = null
            Log.d(TAG, "Stopped recording. File saved to: $outputFilePath")
        } catch (e: Exception) {
            Log.e(TAG, "Error during stop: ${e.message}", e)
        }
    }

    /**
     * Converts an ARGB_8888 Bitmap to an NV12 (YUV420 semi-planar) byte array.
     */
    private fun convertBitmapToYUV(bitmap: Bitmap): ByteArray? {
        try {
            val width = bitmap.width
            val height = bitmap.height
            val frameSize = width * height
            // NV12: Y plane + interleaved UV => total size = frameSize * 3/2
            val yuv = ByteArray(frameSize * 3 / 2)
            val argb = IntArray(frameSize)
            bitmap.getPixels(argb, 0, width, 0, 0, width, height)
            var yIndex = 0
            var uvIndex = frameSize
            for (j in 0 until height) {
                for (i in 0 until width) {
                    val pixel = argb[j * width + i]
                    val r = (pixel shr 16) and 0xff
                    val g = (pixel shr 8) and 0xff
                    val b = pixel and 0xff
                    val yVal = (0.299 * r + 0.587 * g + 0.114 * b).toInt().coerceIn(0, 255)
                    yuv[yIndex++] = yVal.toByte()
                    if (j % 2 == 0 && i % 2 == 0) {
                        val u = ((-0.169 * r - 0.331 * g + 0.5 * b) + 128).toInt().coerceIn(0, 255)
                        val v = ((0.5 * r - 0.419 * g - 0.081 * b) + 128).toInt().coerceIn(0, 255)
                        yuv[uvIndex++] = u.toByte()
                        yuv[uvIndex++] = v.toByte()
                    }
                }
            }
            return yuv
        } catch (e: Exception) {
            Log.e(TAG, "Error converting Bitmap to YUV: ${e.message}", e)
            return null
        }
    }
}
