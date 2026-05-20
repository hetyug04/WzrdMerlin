import UIKit
import SwiftUI
import UniformTypeIdentifiers

// Entry point for the Share Extension.
// Extracts selected text from the host app, then presents the SwiftUI translate UI.
final class ShareViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        extractText { [weak self] text in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.embedSwiftUI(
                    ShareTranslateView(
                        initialText: text,
                        onDismiss: { [weak self] in
                            self?.extensionContext?.completeRequest(returningItems: nil)
                        }
                    )
                )
            }
        }
    }

    // MARK: - Text extraction

    private func extractText(completion: @escaping (String) -> Void) {
        guard
            let item     = extensionContext?.inputItems.first as? NSExtensionItem,
            let provider = item.attachments?.first(where: {
                $0.hasItemConformingToTypeIdentifier(UTType.plainText.identifier)
            })
        else {
            completion("")
            return
        }

        provider.loadItem(forTypeIdentifier: UTType.plainText.identifier) { item, _ in
            completion(item as? String ?? "")
        }
    }

    // MARK: - SwiftUI host

    private func embedSwiftUI(_ rootView: some View) {
        let host = UIHostingController(rootView: rootView)
        addChild(host)
        host.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
        host.didMove(toParent: self)
    }
}
