using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using hAItedetector_UI.Models;

namespace hAItedetector.UI.Controls
{
    public partial class MessageBubble : UserControl
    {
        public MessageBubble()
        {
            InitializeComponent();
        }

        // numele celui care trimite mesajul
        public string SenderName { get; private set; } = "";
        public string CurrentLabel { get; private set; } = "—";
        public string MessageText { get; private set; } = "";

        // ── culori per etichetă ──────────────────────────────────────────
        private static readonly Dictionary<string, (string bg, string fg)> LabelColors = new()
        {
            { "spam",        ("#FF6B00", "#FFFFFF") },
            { "hate_speech", ("#CC0000", "#FFFFFF") },
            { "harassment",  ("#9B0080", "#FFFFFF") },
            { "normal",      ("#1A6B1A", "#FFFFFF") },
            { "eroare",      ("#555555", "#AAAAAA") },
        };

        // ── metodă publică apelată din MainWindow ────────────────────────
        public void SetMessage(
            string text,
            string label,
            string score,
            bool isCensored = false,
            bool alignRight = false,
            string senderName = "")
        {
            if (!string.IsNullOrWhiteSpace(senderName))
            {
                TxtSender.Text = senderName;
                SenderName = senderName;
                CurrentLabel = label;
                MessageText = text;
                TxtSender.Visibility = Visibility.Visible;
            }
            else
            {
                TxtSender.Visibility = Visibility.Collapsed;
            }

            // text — original sau cenzurat
            TxtMessage.Text = text;
            TxtMessage.Foreground = isCensored
                ? new SolidColorBrush(Color.FromRgb(0x88, 0x88, 0x88))
                : new SolidColorBrush(Color.FromRgb(0xD0, 0xD0, 0xD0));

            // etichetă
            TxtLabel.Text = label.ToUpper();
            TxtScore.Text = score != "—" ? $"  {score}" : "";

            // culori badge
            var (bg, fg) = LabelColors.TryGetValue(label, out var c)
                ? c
                : ("#444444", "#AAAAAA");

            LabelBadge.Background = new SolidColorBrush(
                (Color)ColorConverter.ConvertFromString(bg));
            TxtLabel.Foreground = new SolidColorBrush(
                (Color)ColorConverter.ConvertFromString(fg));

            // culoare bulă — mai închisă pentru hate/spam
            BubbleBorder.Background = label switch
            {
                "hate_speech" => new SolidColorBrush(Color.FromRgb(0x3D, 0x10, 0x10)),
                "spam" => new SolidColorBrush(Color.FromRgb(0x3D, 0x25, 0x00)),
                "harassment" => new SolidColorBrush(Color.FromRgb(0x30, 0x10, 0x30)),
                _ => new SolidColorBrush(Color.FromRgb(0x2D, 0x2D, 0x2D)),
            };

            // aliniere stânga/dreapta
            HorizontalAlignment = alignRight
                ? HorizontalAlignment.Right
                : HorizontalAlignment.Left;

            BtnFlag.Visibility = label != "—"
                ? Visibility.Visible
                : Visibility.Collapsed;
        }
        private void BtnFlag_Click(object sender, RoutedEventArgs e)
        {
            // dialog simplu cu ComboBox pentru selectare etichetă corectă
            var dialog = new FlagDialog(CurrentLabel);
            dialog.Owner = Window.GetWindow(this);

            if (dialog.ShowDialog() == true)
            {
                // ridicăm evenimentul către MainWindow
                FlagRequested?.Invoke(this, new FlagEventArgs
                {
                    Text = MessageText,
                    PredictedLabel = CurrentLabel,
                    CorrectLabel = dialog.SelectedLabel,
                });

                // vizual — colorăm butonul ca confirmare
                BtnFlag.Foreground = new SolidColorBrush(
                    (Color)ColorConverter.ConvertFromString("#E8177A")
                );
            }
        }

        public event EventHandler<FlagEventArgs>? FlagRequested;
    }
}