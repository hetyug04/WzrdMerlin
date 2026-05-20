import SwiftUI

struct ShareTranslateView: View {
    let initialText: String
    let onDismiss: () -> Void

    @State private var outputText       = ""
    @State private var isTranslating    = false
    @State private var errorMessage: String?
    @State private var selectedLanguage = SharedDefaults.selectedLanguage
    @State private var didCopy          = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    languagePicker
                    sourcePreview
                    errorBanner
                    outputSection
                }
                .padding()
            }
            .navigationTitle("Translate for Family")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: onDismiss)
                }
            }
            .task { await runTranslation() }
        }
    }

    // MARK: - Subviews

    private var languagePicker: some View {
        HStack {
            Text("Translate to:").foregroundStyle(.secondary)
            Spacer()
            Picker("Language", selection: $selectedLanguage) {
                ForEach(Language.allCases) { lang in
                    Text("\(lang.emoji) \(lang.displayName)").tag(lang)
                }
            }
            .onChange(of: selectedLanguage) { _, new in
                SharedDefaults.selectedLanguage = new
                Task { await runTranslation() }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var sourcePreview: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Original", systemImage: "text.quote")
                .font(.caption).foregroundStyle(.secondary)

            if isTranslating && outputText.isEmpty {
                HStack {
                    ProgressView()
                    Text("Translating…").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            Text(initialText.isEmpty ? "(no text selected)" : initialText)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .lineLimit(5)
        }
    }

    @ViewBuilder
    private var errorBanner: some View {
        if let error = errorMessage {
            VStack(spacing: 10) {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(error).font(.subheadline).frame(maxWidth: .infinity, alignment: .leading)
                }
                Button("Try Again") { Task { await runTranslation() } }
                    .buttonStyle(.bordered)
            }
            .padding()
            .background(Color.orange.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    @ViewBuilder
    private var outputSection: some View {
        if !outputText.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Label(selectedLanguage.displayName, systemImage: "character.bubble.fill")
                    .font(.caption).foregroundStyle(.secondary)

                Text(outputText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                // Primary CTA: copy and return to the host app
                Button(action: copyAndDismiss) {
                    Label(
                        didCopy ? "Copied — paste it now!" : "Copy & Return to App",
                        systemImage: didCopy ? "checkmark.circle.fill" : "doc.on.doc.fill"
                    )
                    .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .tint(didCopy ? .green : .blue)
            }
        }
    }

    // MARK: - Actions

    private func runTranslation() async {
        let text = initialText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        isTranslating = true
        errorMessage  = nil
        outputText    = ""
        didCopy       = false
        do {
            outputText = try await ClaudeService.translate(
                text: text,
                to: selectedLanguage,
                apiKey: SharedDefaults.apiKey
            )
        } catch {
            errorMessage = error.localizedDescription
        }
        isTranslating = false
    }

    private func copyAndDismiss() {
        UIPasteboard.general.string = outputText
        didCopy = true
        Task {
            try? await Task.sleep(for: .seconds(1))
            onDismiss()
        }
    }
}
