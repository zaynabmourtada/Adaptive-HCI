package com.developer27.xamera.openGL2D

import android.app.Activity
import android.app.AlertDialog
import android.os.Bundle
import android.util.Log
import android.widget.EditText

class OpenGL2DActivity : Activity() {

    private lateinit var glView: MyGL2DSurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Create our custom GLSurfaceView
        glView = MyGL2DSurfaceView(this)

        // 2) Set as the content view
        setContentView(glView)

        // 3) Show a message box with an EditText so the user can type a letter/digit
        showCharacterInputDialog()
    }

    // TODO <All Developers>: Inference class must be called here to recieve the input from YOLO (AI) model and be processed for 2D Letters by OpenGL ES.
    /**
     * Displays an AlertDialog with a single EditText.
     * The user can type a character (letter or digit), and we'll draw it.
     */
    private fun showCharacterInputDialog() {
        val editText = EditText(this).apply {
            hint = "Type a letter or digit"
        }

        AlertDialog.Builder(this)
            .setTitle("Enter a Character")
            .setView(editText)
            .setPositiveButton("Draw") { dialog, _ ->
                dialog.dismiss()

                val userText = editText.text.toString().trim()
                if (userText.isNotEmpty()) {
                    // Take the first character
                    val chosenChar = userText[0]
                    Log.d("OpenGLActivity", "User typed: $chosenChar")

                    // Get coordinates for the chosen character
                    val coords = charGenerator.getCoordinatesForChar(chosenChar)

                    // Pass the points to our GLSurfaceView
                    glView.setPoints(coords.toList())
                }
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }
}
