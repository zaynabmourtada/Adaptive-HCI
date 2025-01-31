package com.developer27.xamera.openGL3D

import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent

class MyGL3DSurfaceView(context: Context) : GLSurfaceView(context) {

    private val renderer: MyGL3DRenderer

    // We'll store the user-drawn points here (x, y, z)
    private val userPoints = mutableListOf<Float>()

    // **NEW**: A flag to disable touch if we already have external data
    private var externalPointsSet = false

    init {
        // Request OpenGL ES 2.0
        setEGLContextClientVersion(2)

        // Use our 3D renderer
        renderer = MyGL3DRenderer()
        setRenderer(renderer)

        // Render continuously so user can see updates
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    /**
     * Called by OpenGL3DActivity if "3D_POINTS" were passed from MainActivity.
     * We'll immediately load those points into the renderer and disable user touches.
     */
    fun setExternalPoints(pointsArray: FloatArray) {
        // Clear userPoints and insert the external coords
        userPoints.clear()
        userPoints.addAll(pointsArray.toList())

        // Pass them to the renderer
        renderer.setUserPoints(userPoints)

        // Disable further touch
        externalPointsSet = true
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // If we already have external points from MainActivity, ignore user touch
        if (externalPointsSet) return false

        val x = event.x
        val y = event.y

        // Convert screen coords -> [-1..1] range in X, Y
        val ndcX =  2f * x / width  - 1f
        val ndcY =  1f - 2f * y / height
        // Keep z=0 for a "flat" 3D line
        val z = 0f

        when (event.action) {
            MotionEvent.ACTION_DOWN,
            MotionEvent.ACTION_MOVE -> {
                // Add the new point
                userPoints.add(ndcX)
                userPoints.add(ndcY)
                userPoints.add(z)

                // Send updated points to the renderer
                renderer.setUserPoints(userPoints)
            }
        }
        return true // We handled the touch
    }
}