using hAItedetector.UI.Controls;
using hAItedetector_UI.Models;
using hAItedetector_UI.Services;
using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media;

namespace hAItedetector.UI
{
    public partial class MainWindow : Window
    {
        private string _currentUser = "";
        private readonly ApiService _api = new();
        private readonly ObservableCollection<MessageItem> _messages = new();
        private bool _censorHate = false;
        private string? _lang = null;

        public MainWindow()
        {
            InitializeComponent();
        }

        // ── pornire fereastră → verificăm serverul ──────────────────────
        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            await CheckServerAsync();
        }

        private async Task CheckServerAsync()
        {
            TxtStatus.Text = "Server is starting...";

            // încercăm de 10 ori cu pauză de 2 secunde între încercări
            // total: până la 20 secunde — suficient pentru încărcarea modelului
            for (int i = 0; i < 10; i++)
            {
                await Task.Delay(2000);

                bool alive = await _api.IsServerAliveAsync();

                if (alive)
                {
                    StatusDot.Fill = new SolidColorBrush(Colors.Green);
                    TxtStatus.Text = "Server online";
                    return;
                }

                TxtStatus.Text = $"Server is starting... ({i + 1}/10)";
            }

            // dacă după 20 secunde tot nu răspunde
            StatusDot.Fill = new SolidColorBrush(Colors.Red);
            TxtStatus.Text = "Server offline";
        }

        // ── Import ───────────────────────────────────────────────────
        private void BtnImport_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Filter = "Supported files (*.csv;*.txt)|*.csv;*.txt|CSV (*.csv)|*.csv|WhatsApp Export (*.txt)|*.txt|All files (*.*)|*.*",
                Title = "Select exported conversation file",
            };

            if (dialog.ShowDialog() != true) return;

            MessagesPanel.Children.Clear();
            _messages.Clear();
            _currentUser = "";

            if (dialog.FileName.EndsWith(".txt", StringComparison.OrdinalIgnoreCase))
                ParseWhatsApp(dialog.FileName);
            else
                ParseCsv(dialog.FileName);
        }
        private void ParseCsv(string filePath)
        {
            var lines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);
            bool isFirst = true;
            int textColumnIndex = 2;
            int senderColumnIndex = -1;

            foreach (var line in lines)
            {
                if (isFirst)
                {
                    isFirst = false;
                    var headers = ParseCsvLine(line);
                    for (int i = 0; i < headers.Count; i++)
                    {
                        var h = headers[i].ToLower();
                        if (h == "text" || h == "body" || h == "sms" || h == "message")
                            textColumnIndex = i;
                        if (h == "sender" || h == "from" || h == "address")
                            senderColumnIndex = i;
                    }
                    continue;
                }

                if (string.IsNullOrWhiteSpace(line)) continue;

                var cols = ParseCsvLine(line);
                if (cols.Count <= textColumnIndex) continue;

                var text = string.Join(",", cols.Skip(textColumnIndex)).Trim();
                var msgSender = senderColumnIndex >= 0 && cols.Count > senderColumnIndex
                                ? cols[senderColumnIndex].Trim()
                                : "";

                if (string.IsNullOrWhiteSpace(text)) continue;

                if (string.IsNullOrWhiteSpace(_currentUser) &&
                    !string.IsNullOrWhiteSpace(msgSender))
                    _currentUser = msgSender;

                bool alignRight = !string.IsNullOrWhiteSpace(msgSender) &&
                                  msgSender == _currentUser;

                _messages.Add(new MessageItem { Text = text });

                var bubble = new Controls.MessageBubble();
                bubble.SetMessage(
                    text: text,
                    label: "—",
                    score: "—",
                    alignRight: alignRight,
                    senderName: msgSender
                );
                MessagesPanel.Children.Add(bubble);
            }

            BtnAnalyze.IsEnabled = _messages.Count > 0;
            TxtStatus.Text = $"{_messages.Count} imported messages (CSV)";
        }

        // Parser CSV corect — respectă câmpurile între ghilimele
        private static List<string> ParseCsvLine(string line)
        {
            var result = new List<string>();
            var current = new System.Text.StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];

                if (c == '"')
                {
                    // ghilimele duble în interiorul unui câmp între ghilimele → escape
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                }
                else if (c == ',' && !inQuotes)
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(c);
                }
            }

            result.Add(current.ToString());
            return result;
        }

        // Parser import WhatsApp
        private void ParseWhatsApp(string filePath)
        {
            MessagesPanel.Children.Clear();
            _messages.Clear();
            _currentUser = "";

            var lines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);

            // pattern: [16.06.2025, 21:27:11] Expeditor: Mesaj
            var messagePattern = new System.Text.RegularExpressions.Regex(
                @"^\[(\d{1,2}\.\d{1,2}\.\d{4}), \d{2}:\d{2}:\d{2}\] ([^:]+): (.+)$"
            );

            string currentSender = "";
            string currentMessage = "";

            void FlushMessage()
            {
                if (string.IsNullOrWhiteSpace(currentMessage)) return;

                var text = currentMessage.Trim();

                // ignorăm mesajele sistem WhatsApp
                if (text.Contains("Messages and calls are end-to-end encrypted.") ||
                    text.Contains("changed their phone number. You're currently chatting with their new number") ||
                    text.Contains("Mesajele și apelurile sunt criptate integral.") ||
                    text.Contains("și-a schimbat numărul de telefon. Această conversație se desfășoară cu noul număr") ||
                    text.EndsWith("document omis") ||
                    text.EndsWith("document omitted") ||
                    text.EndsWith("imagine omisă") ||
                    text.EndsWith("image omitted") ||
                    text.EndsWith("video omis") ||
                    text.EndsWith("video omitted") ||
                    text.EndsWith("audio omis") ||
                    text.EndsWith("audio omitted") ||
                    text.EndsWith("GIF omis") ||
                    text.EndsWith("sticker omis")) {
                    currentSender = "";
                    currentMessage = "";
                    return;
                }

                if (string.IsNullOrWhiteSpace(_currentUser) &&
                    !string.IsNullOrWhiteSpace(currentSender))
                    _currentUser = currentSender;

                bool alignRight = !string.IsNullOrWhiteSpace(currentSender) &&
                                  currentSender == _currentUser;

                _messages.Add(new MessageItem { Text = text });

                var bubble = new Controls.MessageBubble();
                bubble.SetMessage(text, "—", "—",
                                  alignRight: alignRight,
                                  senderName: currentSender);
                MessagesPanel.Children.Add(bubble);

                currentSender = "";
                currentMessage = "";
            }

            foreach (var line in lines)
            {
                // stripuim caracterele invizibile Unicode pe care WhatsApp le inserează
                // U+200E (Left-to-Right Mark), U+200F (Right-to-Left Mark), U+FEFF (BOM)
                var cleanLine = line
                    .Replace("\u200E", "")
                    .Replace("\u200F", "")
                    .Replace("\uFEFF", "")
                    .Trim();

                var match = messagePattern.Match(cleanLine);

                if (match.Success)
                {
                    // salvăm mesajul anterior înainte să începem unul nou
                    FlushMessage();

                    currentSender = match.Groups[2].Value.Trim();
                    currentMessage = match.Groups[3].Value;
                }
                else if (!string.IsNullOrWhiteSpace(currentMessage))
                {
                    // continuare mesaj pe linia următoare
                    currentMessage += " " + cleanLine.Trim();
                }
            }

            // ultimul mesaj
            FlushMessage();

            BtnAnalyze.IsEnabled = _messages.Count > 0;
            TxtStatus.Text = $"{_messages.Count} imported messages (WhatsApp)";
        }

        // ── Analyze ──────────────────────────────────────────────────────
        private async void BtnAnalyze_Click(object sender, RoutedEventArgs e)
        {
            BtnAnalyze.IsEnabled = false;
            TxtStatus.Text = "Analyzing...";

            // citim setarea de cenzură din Settings
            bool censorHate = _censorHate;

            for (int i = 0; i < _messages.Count; i++)
            {
                var item = _messages[i];

                try
                {
                    var result = await _api.DetectAsync(item.Text, lang: _lang, censorHate: censorHate);
                    if (result == null) continue;

                    string label = "normal";
                    double score = 0;

                    foreach (var (key, val) in result.Decisions)
                    {
                        if (val && result.Scores.TryGetValue(key, out double s) && s > score)
                        {
                            label = key;
                            score = s;
                        }
                    }

                    string displayText = censorHate && result.CensoredText != null
                        ? result.CensoredText
                        : item.Text;

                    // actualizăm bula existentă
                    if (MessagesPanel.Children[i] is Controls.MessageBubble bubble)
                    {
                        // recuperăm senderul din bula existentă
                        bool alignRight = bubble.SenderName == _currentUser;

                        bubble.SetMessage(
                            text: displayText,
                            label: label,
                            score: score > 0 ? score.ToString("F2") : "—",
                            isCensored: result.CensoredText != null,
                            alignRight: alignRight,
                            senderName: bubble.SenderName
                        );

                        // actualizare nr mesaje analizate
                        _messages[i] = new MessageItem
                        {
                            Text = displayText,
                            Label = label,
                            Score = score > 0 ? score.ToString("F2") : "—",
                            Scores = result.Scores,
                            SenderName = bubble.SenderName,
                        };

                        // abonare la evenimentul de flag
                        bubble.FlagRequested -= OnFlagRequested; // evităm dubluri
                        bubble.FlagRequested += OnFlagRequested;
                    }
                }
                catch
                {
                    if (MessagesPanel.Children[i] is Controls.MessageBubble bubble)
                        bubble.SetMessage(item.Text, "error", "—");
                }

                TxtStatus.Text = $"Analyzed {i + 1} / {_messages.Count}...";
            }

            BtnAnalyze.IsEnabled = true;
            TxtStatus.Text = $"Done — {_messages.Count} analyzed messages";
        }

        // Flag request
        private async void OnFlagRequested(object? sender, FlagEventArgs e)
        {
            bool saved = await _api.FlagMessageAsync(
                e.Text,
                e.PredictedLabel,
                e.CorrectLabel
            );

            TxtStatus.Text = saved
                ? $"Reported: '{(e.Text.Length > 30 ? e.Text[..30] + "..." : e.Text)}' → {e.CorrectLabel}"
                : "Error at saving reported message.";
        }

        // Export
        private void BtnExport_Click(object sender, RoutedEventArgs e)
        {
            var analyzed = _messages
                .Where(m => m.Label != "—")
                .ToList();

            if (!analyzed.Any())
            {
                TxtStatus.Text = "No analyzed messages to export.";
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                FileName = $"haitedetector_export_{DateTime.Now:yyyyMMdd_HHmmss}.csv",
            };

            if (dialog.ShowDialog() != true) return;

            using var writer = new StreamWriter(dialog.FileName, false, System.Text.Encoding.UTF8);
            writer.WriteLine("text,label,score,spam,hate_speech,harassment,obscene,sender");

            foreach (var msg in analyzed)
            {
                var text = $"\"{msg.Text.Replace("\"", "\"\"")}\"";
                var label = msg.Label;
                var score = msg.Score;
                var spamScore = msg.Scores.TryGetValue("spam", out double sp) ? sp.ToString("F4", CultureInfo.InvariantCulture) : "—";
                var hateScore = msg.Scores.TryGetValue("hate_speech", out double hs) ? hs.ToString("F4", CultureInfo.InvariantCulture) : "—";
                var harassScore = msg.Scores.TryGetValue("harassment", out double hr) ? hr.ToString("F4", CultureInfo.InvariantCulture) : "—";
                var obsceneScore = msg.Scores.TryGetValue("obscene", out double ob) ? ob.ToString("F4", CultureInfo.InvariantCulture) : "—";
                var msgSender = $"\"{msg.SenderName?.Replace("\"", "\"\"") ?? ""}\"";
                writer.WriteLine($"{text},{label},{score},{spamScore},{hateScore},{harassScore},{obsceneScore},{msgSender}");
            }

            TxtStatus.Text = $"Exported {analyzed.Count} messages in {Path.GetFileName(dialog.FileName)}";
        }

        // ── Settings ─────────────────────────────────────────────────────
        private void BtnSettings_Click(object sender, RoutedEventArgs e)
        {
            var settings = new SettingsWindow(_censorHate, _lang);
            settings.Owner = this;

            if (settings.ShowDialog() == true)
            {
                _censorHate = settings.CensorHate;
                _lang = settings.Lang;
            }
        }
    }
}