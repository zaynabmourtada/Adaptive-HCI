// File: OpenGLRenderer.kt
package com.developer27.xamera

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.opengl.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.nio.FloatBuffer
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator

// TODO <Soham Naik>: Test this code whether it is working as it is. Otherwise, make adjustments.
/**
 * Simplified OpenGLRenderer for rendering pre-filtered and post-filtered paths.
 * It initializes OpenGL ES 2.0, compiles shaders, sets up VBOs, and renders the paths.
 */
class OpenGLRenderer(private val videoProcessor: VideoProcessor) {

    // EGL context-related members
    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null

    // Shader program and handles
    private var shaderProgram: Int = 0
    private var positionHandle: Int = 0
    private var colorHandle: Int = 0
    private var mvpHandle: Int = 0

    // VBOs for pre-filtered and post-filtered paths
    private var preFilterVBO: Int = 0
    private var postFilterVBO: Int = 0

    // Vertex counts
    private var preFilterVertexCount: Int = 0
    private var postFilterVertexCount: Int = 0

    // Colors for the paths
    private val preFilterColor = floatArrayOf(0f, 0f, 1f, 1f)    // Blue
    private val postFilterColor = floatArrayOf(1f, 1f, 0f, 1f)   // Yellow

    // Transformation matrices
    private val mvpMatrix = FloatArray(16)
    private val projectionMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)

    // Coroutine scope for asynchronous tasks
    private val rendererScope = CoroutineScope(Dispatchers.Main + Job())

    // Handler for frame rendering
    private val handler = Handler(Looper.getMainLooper())

    /**
     * Starts the renderer by setting up EGL, initializing OpenGL resources,
     * handling surface changes, and entering the render loop.
     */
    fun start(surfaceTexture: SurfaceTexture, width: Int, height: Int) {
        setupEGL(surfaceTexture)
        initializeOpenGL()
        onSurfaceChanged(width, height)
        renderLoop()
    }

    /**
     * Sets up the EGL context and surface.
     */
    private fun setupEGL(surfaceTexture: SurfaceTexture) {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw RuntimeException("Unable to get EGL14 display")
        }

        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw RuntimeException("Unable to initialize EGL14")
        }

        val eglConfigAttributes = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )

        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(eglDisplay, eglConfigAttributes, 0, configs, 0, configs.size, numConfigs, 0)) {
            throw RuntimeException("Unable to choose EGL config")
        }

        val eglContextAttributes = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, eglContextAttributes, 0)

        val eglSurfaceAttributes = intArrayOf(EGL14.EGL_NONE)
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], surfaceTexture, eglSurfaceAttributes, 0)

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw RuntimeException("Failed to make EGL context current")
        }
    }

    /**
     * Initializes OpenGL resources like shaders and VBOs.
     */
    private fun initializeOpenGL() {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        shaderProgram = createShaderProgram()
        GLES20.glUseProgram(shaderProgram)

        positionHandle = GLES20.glGetAttribLocation(shaderProgram, "vPosition")
        colorHandle = GLES20.glGetUniformLocation(shaderProgram, "vColor")
        mvpHandle = GLES20.glGetUniformLocation(shaderProgram, "uMVPMatrix")

        Log.d("OpenGLRenderer", "Shader program initialized. Handles - Position: $positionHandle, Color: $colorHandle, MVP: $mvpHandle")

        val vbos = IntArray(2)
        GLES20.glGenBuffers(2, vbos, 0)
        preFilterVBO = vbos[0]
        postFilterVBO = vbos[1]

        Log.d("OpenGLRenderer", "VBOs initialized. preFilterVBO: $preFilterVBO, postFilterVBO: $postFilterVBO")

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, preFilterVBO)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, MAX_PATH_POINTS * 3 * 4, null, GLES20.GL_DYNAMIC_DRAW)
        checkGLError("initializeOpenGL - preFilterVBO")

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, postFilterVBO)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, MAX_PATH_POINTS * 3 * 4, null, GLES20.GL_DYNAMIC_DRAW)
        checkGLError("initializeOpenGL - postFilterVBO")

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
    }

    /**
     * Adjusts the viewport and projection matrix when the surface size changes.
     */
    fun onSurfaceChanged(width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val aspectRatio = width.toFloat() / height
        Matrix.frustumM(projectionMatrix, 0, -aspectRatio, aspectRatio, -1f, 1f, 3f, 7000f)
    }

    /**
     * The main rendering loop, scheduled on the main thread with frame rate control.
     */
    private fun renderLoop() {
        val targetFps = 60
        val frameDuration = 1000 / targetFps

        val renderRunnable = object : Runnable {
            override fun run() {

                val startTime = System.currentTimeMillis()

                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

                onDrawFrame()

                EGL14.eglSwapBuffers(eglDisplay, eglSurface)

                val elapsedTime = System.currentTimeMillis() - startTime
                val delay = (frameDuration - elapsedTime).coerceAtLeast(0)
                handler.postDelayed(this, delay)
            }
        }

        handler.post(renderRunnable)
    }

    /**
     * Handles drawing each frame, including pre-filtered and post-filtered paths.
     */
    private fun onDrawFrame() {
        // Set up the camera view
        Matrix.setLookAtM(viewMatrix, 0,
            0f, 0f, -1000f,  // Camera position
            0f, 0f, 0f,      // Look at point
            0f, 1f, 0f       // Up vector
        )

        // Identity model matrix (no transformations)
        Matrix.setIdentityM(modelMatrix, 0)

        // Calculate the Model-View-Projection matrix
        Matrix.multiplyMM(mvpMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, mvpMatrix, 0)

        // Launch a coroutine to update VBOs with new data
        rendererScope.launch {
            try {
                // Retrieve pre-filtered and post-filtered data
                val preFilterData = videoProcessor.retrievePreFilter4Ddata()
                val postFilterData = videoProcessor.retrievePostFilter4Ddata()

                Log.d("OpenGLRenderer", "Pre-filtered data size: ${preFilterData.size}")
                Log.d("OpenGLRenderer", "Post-filtered data size: ${postFilterData.size}")

                // Update VBOs
                preFilterVertexCount = updateVBO(preFilterVBO, preFilterData)
                postFilterVertexCount = updateVBO(postFilterVBO, postFilterData)
            } catch (e: Exception) {
                Log.e("OpenGLRenderer", "Error updating VBOs: ${e.message}")
                e.printStackTrace()
            }
        }

        // Draw pre-filtered path
        drawPath(preFilterVBO, preFilterVertexCount, preFilterColor, mvpMatrix)

        // Draw post-filtered path
        drawPath(postFilterVBO, postFilterVertexCount, postFilterColor, mvpMatrix)
    }

    private fun checkGLError(operation: String) {
        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            Log.e("OpenGLRenderer", "$operation: glError $error")
            throw RuntimeException("$operation: glError $error")
        }
    }

    /**
     * Updates a VBO with the provided FrameData.
     * @param vbo The Vertex Buffer Object to update.
     * @param data The list of FrameData to render.
     * @return The number of vertices updated.
     */
    private fun updateVBO(vbo: Int, data: List<FrameData>): Int {
        val vertexCount = minOf(data.size, MAX_PATH_POINTS)
        val buffer = FloatArray(vertexCount * 3)

        for (i in 0 until vertexCount) {
            val frame = data[i]
            buffer[i * 3] = frame.x.toFloat()
            buffer[i * 3 + 1] = frame.y.toFloat()
            buffer[i * 3 + 2] = 0f
        }

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
        GLES20.glBufferSubData(GLES20.GL_ARRAY_BUFFER, 0, vertexCount * 3 * 4, FloatBuffer.wrap(buffer, 0, vertexCount * 3))
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)

        Log.d("OpenGLRenderer", "Updated VBO $vbo with $vertexCount vertices.")
        checkGLError("updateVBO")

        return vertexCount
    }

    /**
     * Draws a path using a Vertex Buffer Object (VBO).
     * @param vbo The Vertex Buffer Object containing path vertices.
     * @param count The number of vertices to draw.
     * @param color The color of the path.
     * @param mvpMatrix The Model-View-Projection matrix.
     */
    private fun drawPath(vbo: Int, count: Int, color: FloatArray, mvpMatrix: FloatArray) {
        if (count == 0) return

        GLES20.glUseProgram(shaderProgram)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(
            positionHandle, 3, GLES20.GL_FLOAT, false, 3 * 4, 0
        )

        GLES20.glUniform4fv(colorHandle, 1, color, 0)
        GLES20.glUniformMatrix4fv(mvpHandle, 1, false, mvpMatrix, 0)

        GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, count)

        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
    }

    /**
     * Creates and compiles the shader program, then links vertex and fragment shaders.
     * @return The OpenGL shader program ID.
     */
    private fun createShaderProgram(): Int {
        val vertexShaderCode = """
            uniform mat4 uMVPMatrix;
            attribute vec4 vPosition;
            void main() {
                gl_Position = uMVPMatrix * vPosition;
            }
        """.trimIndent()

        val fragmentShaderCode = """
            precision mediump float;
            uniform vec4 vColor;
            void main() {
                gl_FragColor = vColor;
            }
        """.trimIndent()

        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        return GLES20.glCreateProgram().also { program ->
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)

            // Check linking status
            val linkStatus = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] == 0) {
                Log.e("ShaderProgram", "Error linking program: ${GLES20.glGetProgramInfoLog(program)}")
                GLES20.glDeleteProgram(program)
                throw RuntimeException("Error creating shader program.")
            }
        }
    }

    /**
     * Compiles a shader of the given type with the provided source code.
     * @param type The type of shader (vertex or fragment).
     * @param shaderCode The GLSL code of the shader.
     * @return The OpenGL shader ID.
     */
    private fun loadShader(type: Int, shaderCode: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)

            // Check compile status
            val compileStatus = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compileStatus, 0)
            if (compileStatus[0] == 0) {
                Log.e("Shader", "Error compiling shader: ${GLES20.glGetShaderInfoLog(shader)}")
                GLES20.glDeleteShader(shader)
                throw RuntimeException("Error compiling shader.")
            }
        }
    }

    /**
     * Stops the rendering loop and cancels coroutines.
     */
    fun stop() {
        rendererScope.cancel()
    }

    /**
     * Releases EGL and OpenGL resources.
     */
    fun release() {
        if (eglDisplay != null && eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(
                eglDisplay, EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT
            )
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)
        }

        eglDisplay = null
        eglContext = null
        eglSurface = null

        // Delete VBOs
        val vbos = intArrayOf(preFilterVBO, postFilterVBO)
        GLES20.glDeleteBuffers(2, vbos, 0)
    }

    companion object {
        private const val MAX_PATH_POINTS = 1000 // Maximum number of points per path
    }
}
