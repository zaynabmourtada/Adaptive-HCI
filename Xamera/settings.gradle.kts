pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // ARCore plugin repository (sometimes needed):
        maven {
            url = uri("https://android.arcore.google.com")
        }
    }
}

rootProject.name = "Xamera"
include(":app")
include(":OpenCV-4.10.0")