using hAItedetector_UI.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace hAItedetector_UI.Services
{
    public class ApiService
    {
        private readonly HttpClient _client;
        private readonly string _baseUrl;

        // JsonSerializerOptions refolosit — evită alocări repetate
        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true,
        };

        public ApiService(string baseUrl = "http://127.0.0.1:8000")
        {
            _baseUrl = baseUrl;
            _client = new HttpClient
            {
                // timeout generous pentru primul request
                // (modelul DistilBERT se încarcă la primul apel)
                Timeout = TimeSpan.FromSeconds(60),
            };
        }

        // -------------------------------------------------------------------
        // Health check — verifică dacă serverul Python rulează
        // -------------------------------------------------------------------

        public async Task<bool> IsServerAliveAsync()
        {
            try
            {
                var response = await _client.GetAsync($"{_baseUrl}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                // orice excepție (connection refused, timeout) → server offline
                return false;
            }
        }

        // -------------------------------------------------------------------
        // Detect — trimite textul și primește rezultatul
        // -------------------------------------------------------------------

        public async Task<DetectResponse?> DetectAsync(
            string text,
            string? lang = null,
            bool censorHate = false)
        {
            var request = new DetectRequest
            {
                Text = text,
                Lang = lang,
                CensorHate = censorHate,
            };

            // serializăm request-ul în JSON
            var json = JsonSerializer.Serialize(request, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            try
            {
                var response = await _client.PostAsync($"{_baseUrl}/detect", content);

                if (!response.IsSuccessStatusCode)
                {
                    var error = await response.Content.ReadAsStringAsync();
                    throw new Exception($"API error {response.StatusCode}: {error}");
                }

                var responseJson = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<DetectResponse>(responseJson, _jsonOptions);
            }
            catch (TaskCanceledException)
            {
                throw new Exception("Server timeout.");
            }
            catch (HttpRequestException)
            {
                throw new Exception("Cannot connect to server. Try running run.py manually.");
            }
        }

        public async Task<bool> FlagMessageAsync(string text, string predictedLabel, string correctLabel)
        {
            var payload = new
            {
                text = text,
                predicted_label = predictedLabel,
                correct_label = correctLabel,
            };

            var json = JsonSerializer.Serialize(payload, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            try
            {
                var response = await _client.PostAsync($"{_baseUrl}/flag", content);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
    }
}