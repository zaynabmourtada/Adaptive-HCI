package com.developer27.xamera.openGL3D

import android.app.Activity
import android.os.Bundle

class OpenGL3DActivity : Activity() {

    private lateinit var glView: MyGL3DSurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create our custom GLSurfaceView that can handle user touches
        glView = MyGL3DSurfaceView(this)

        // **NEW**: Check if we have "3D_POINTS" from the Intent
        val pointsArray = intent.getFloatArrayExtra("3D_POINTS")
        if (pointsArray != null) {
            if (!pointsArray.isEmpty()) {
                // Pass these coordinates to glView, so it draws them (instead of user touch)
                glView.setExternalPoints(pointsArray)
            }
        }

        setContentView(glView)
    }
}