using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Windows;

namespace hAItedetector_UI
{
    public partial class App : Application
    {
        private Process? _pythonProcess;

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            StartPythonServer();
        }

        protected override void OnExit(ExitEventArgs e)
        {
            StopPythonServer();
            base.OnExit(e);
        }

        private string FindProjectRoot()
        {
            var dir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
            while (dir != null)
            {
                // caută folderul backend cu venv și run.py în el
                var backendPath = Path.Combine(dir.FullName, "backend");
                if (Directory.Exists(Path.Combine(backendPath, "venv")) &&
                    File.Exists(Path.Combine(backendPath, "run.py")))
                    return backendPath; // returnează direct backend/
                dir = dir.Parent;
            }
            throw new DirectoryNotFoundException("Nu s-a găsit folderul backend.");
        }

        private void StartPythonServer()
        {
            string backendRoot;
            try { backendRoot = FindProjectRoot(); }
            catch (DirectoryNotFoundException ex)
            {
                MessageBox.Show(ex.Message, "hAItedetector — Error",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var pythonExe = Path.Combine(backendRoot, @"venv\Scripts\python.exe");
            var runScript = Path.Combine(backendRoot, "run.py");

            // verificăm că fișierele există
            if (!File.Exists(pythonExe) || !File.Exists(runScript))
            {
                MessageBox.Show(
                    $"Python server not found.\nSearch:\n{pythonExe}\n{runScript}",
                    "hAItedetector — Error",
                    MessageBoxButton.OK,
                    MessageBoxImage.Warning
                );
                return;
            }

            _pythonProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{runScript}\"",
                    WorkingDirectory = backendRoot,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                }
            };

            _pythonProcess.OutputDataReceived += (s, e) => {
                if (e.Data != null) Debug.WriteLine($"[Python OUT] {e.Data}");
            };
            _pythonProcess.ErrorDataReceived += (s, e) => {
                if (e.Data != null) Debug.WriteLine($"[Python ERR] {e.Data}");
            };

            _pythonProcess.Start();
            _pythonProcess.BeginOutputReadLine();
            _pythonProcess.BeginErrorReadLine();
        }

        private void StopPythonServer()
        {
            try
            {
                if (_pythonProcess != null && !_pythonProcess.HasExited)
                {
                    _pythonProcess.Kill(entireProcessTree: true);
                    _pythonProcess.Dispose();
                }
            }
            catch { /* ignorăm erorile la oprire */ }
        }
    }
}