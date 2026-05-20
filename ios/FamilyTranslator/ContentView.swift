import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            TranslateView()
                .tabItem { Label("Translate", systemImage: "globe") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
    }
}
