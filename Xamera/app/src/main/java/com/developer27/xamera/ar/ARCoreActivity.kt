package com.developer27.xamera.ar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Session

class ARCoreActivity : AppCompatActivity() {
    private var arCoreSession: Session? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Check if ARCore is installed & up to date
        val availability = ArCoreApk.getInstance().checkAvailability(this)
        if (availability.isTransient) {
            // Check again later...
        } else if (availability.isSupported) {
            // The device supports AR
        } else {
            // Not supported
            Toast.makeText(this, "ARCore not supported on this device.", Toast.LENGTH_LONG).show()
            finish()
        }

        // 2) Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE
            )
        } else {
            // If we already have permission, set up ARCore session
            setupSession()
        }
    }

    private fun setupSession() {
        try {
            arCoreSession = Session( /* context= */this)
            // Configure the session if needed
            // Session.Config config = new Session.Config(arCoreSession);
            // arCoreSession.configure(config);
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error creating ARCore session: " + e.message, Toast.LENGTH_LONG)
                .show()
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        if (arCoreSession != null) {
            try {
                arCoreSession!!.resume()
                // Then attach session to your renderer or camera preview if you have one.
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (arCoreSession != null) {
            // Pause the session to save resources
            arCoreSession!!.pause()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (arCoreSession != null) {
            arCoreSession!!.close()
            arCoreSession = null
        }
    }

    // Handle camera permission result
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.size > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED
            ) {
                setupSession()
            } else {
                Toast.makeText(this, "Camera permission is required for AR", Toast.LENGTH_LONG)
                    .show()
                finish()
            }
        }
    }

    companion object {
        private const val CAMERA_PERMISSION_CODE = 101
    }
}