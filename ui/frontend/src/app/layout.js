import './globals.css';

export const metadata = {
  title: 'DeepFake Detection',
  description: 'Upload a video and get a real-time deepfake verdict with explainability.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
