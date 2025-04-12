package com.developer27.xamera

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.ListPreference
import androidx.preference.PreferenceFragmentCompat
import com.developer27.xamera.videoprocessing.Settings

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.settings_activity)

        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // Set up the detection mode preference listener.
            val detectionModePref = findPreference<ListPreference>("detection_mode")
            detectionModePref?.setOnPreferenceChangeListener { _, newValue ->
                when (newValue as String) {
                    "CONTOUR" -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
                        Settings.DetectionMode.enableYOLOinference = false
                    }
                    "YOLO" -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.YOLO
                        Settings.DetectionMode.enableYOLOinference = true
                    }
                    else -> {
                        Settings.DetectionMode.current = Settings.DetectionMode.Mode.CONTOUR
                        Settings.DetectionMode.enableYOLOinference = false
                    }
                }
                Toast.makeText(context, "Detection mode set to $newValue", Toast.LENGTH_SHORT).show()
                true
            }
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        setResult(RESULT_OK, Intent())
        finish()
    }
}