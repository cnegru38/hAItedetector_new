using System.Windows;
using System.Windows.Controls;

namespace hAItedetector.UI
{
    public partial class SettingsWindow : Window
    {
        // proprietăți publice citite din MainWindow după ShowDialog()
        public bool CensorHate { get; private set; } = false;
        public string? Lang { get; private set; } = null;

        public SettingsWindow(bool currentCensor, string? currentLang)
        {
            InitializeComponent();

            // populăm cu valorile curente
            ChkCensor.IsChecked = currentCensor;

            // selectăm limba curentă în ComboBox
            foreach (ComboBoxItem item in CmbLang.Items)
            {
                var tag = item.Tag as string;
                if (tag == currentLang)
                {
                    item.IsSelected = true;
                    break;
                }
            }
        }

        private void BtnSave_Click(object sender, RoutedEventArgs e)
        {
            CensorHate = ChkCensor.IsChecked == true;

            // extragem tag-ul din item-ul selectat
            if (CmbLang.SelectedItem is ComboBoxItem selected)
                Lang = selected.Tag as string;  // null = Auto

            DialogResult = true;  // semnalează MainWindow că s-a salvat
            Close();
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }

        private void BtnAbout_Click(object sender, RoutedEventArgs e)
        {
            var about = new AboutWindow();
            about.Owner = this;
            about.ShowDialog();
        }
    }
}