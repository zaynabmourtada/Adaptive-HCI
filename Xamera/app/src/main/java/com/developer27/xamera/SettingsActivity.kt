package com.developer27.xamera

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.PreferenceManager

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

            // Get preferences and add change listeners
            val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(requireContext())
            val frequencyPreference: ListPreference? = findPreference("rolling_shutter_frequency")

            frequencyPreference?.onPreferenceChangeListener = Preference.OnPreferenceChangeListener { _, newValue ->
                // Display a toast message to indicate the frequency change
                val newFrequency = newValue.toString()
                Toast.makeText(requireContext(), "Rolling shutter frequency set to $newFrequency Hz", Toast.LENGTH_SHORT).show()

                // Notify SettingsActivity that preferences have changed
                activity?.setResult(RESULT_OK, Intent().apply { putExtra("settingsChanged", true) })
                true
            }
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        // Save settings and go back to MainActivity with result
        val resultIntent = Intent()
        resultIntent.putExtra("settingsChanged", true)
        setResult(RESULT_OK, resultIntent)
        finish()
    }
}