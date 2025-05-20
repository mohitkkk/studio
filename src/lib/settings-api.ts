import { ModelSettings, FormatSettings, NoInfoSettings } from "./settings-types";

// Define types if not already defined in settings-types.ts
export type ChatUploadSettings = {
  processOnUpload: boolean;
  autoAddToChat: boolean;
  maxFileSize: number; // in MB
  allowedFileTypes: string[];
};

// Default upload settings
const defaultUploadSettings: ChatUploadSettings = {
  processOnUpload: true,
  autoAddToChat: true,
  maxFileSize: 10, // 10MB
  allowedFileTypes: [".txt", ".pdf", ".md", ".docx", ".json"],
};

// Load upload settings
export function loadUploadSettings(): ChatUploadSettings {
  if (typeof window === 'undefined') return defaultUploadSettings;
  
  const savedSettings = localStorage.getItem("chatbot_upload_settings");
  if (!savedSettings) return defaultUploadSettings;
  
  try {
    return { ...defaultUploadSettings, ...JSON.parse(savedSettings) };
  } catch (error) {
    console.error("Error parsing upload settings:", error);
    return defaultUploadSettings;
  }
}

// Save upload settings
export function saveUploadSettings(settings: ChatUploadSettings): void {
  localStorage.setItem("chatbot_upload_settings", JSON.stringify(settings));
}

// Get all settings combined for chat API requests
export function getChatRequestSettings(selectedFiles: string[], clientId: string) {
  // Load all settings
  const uploadSettings = loadUploadSettings();
  
  // Load other settings (assuming these functions are implemented)
  const modelSettings = JSON.parse(localStorage.getItem("chatbot_model_settings") || "{}");
  const formatSettings = JSON.parse(localStorage.getItem("chatbot_format_settings") || "{}");
  const noInfoSettings = JSON.parse(localStorage.getItem("chatbot_noinfo_settings") || "{}");
  
  // Build unified request object
  return {
    selected_files: selectedFiles,
    client_id: clientId,
    // Apply model settings
    ollama_model: modelSettings.ollama_model || "llama3",
    ollama_embedding_model: modelSettings.ollama_embedding_model || "mxbai-embed-large",
    temperature: modelSettings.temperature ?? 0.7,
    top_p: modelSettings.top_p ?? 1.0,
    top_k_per_file: modelSettings.top_k_per_file ?? 5,
    similarity_threshold: modelSettings.similarity_threshold ?? 0.5,
    system_prompt: modelSettings.system_prompt || "",
    // Apply format settings
    format: formatSettings.format || "html",
    clean_response: formatSettings.clean_response ?? true,
    remove_hedging: formatSettings.remove_hedging ?? true,
    remove_references: formatSettings.remove_references ?? true,
    remove_disclaimers: formatSettings.remove_disclaimers ?? true,
    ensure_html_structure: formatSettings.ensure_html_structure ?? true,
    custom_hedging_patterns: formatSettings.custom_hedging_patterns || [],
    custom_reference_patterns: formatSettings.custom_reference_patterns || [],
    custom_disclaimer_patterns: formatSettings.custom_disclaimer_patterns || [],
    html_tags_to_fix: formatSettings.html_tags_to_fix || [],
    // Apply no-info settings
    no_info_title: noInfoSettings.no_info_title || "Information Not Found",
    no_info_message: noInfoSettings.no_info_message || "I couldn't find specific information about that in the selected document(s).",
    include_suggestions: noInfoSettings.include_suggestions ?? true,
    custom_suggestions: noInfoSettings.custom_suggestions || [],
    no_info_html_format: noInfoSettings.no_info_html_format ?? true,
  };
}