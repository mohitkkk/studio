export type ModelSettings = {
  ollama_model: string;
  ollama_embedding_model: string;
  temperature: number;
  top_p: number;
  top_k_per_file: number;
  similarity_threshold: number;
  system_prompt: string;
};

export type FormatSettings = {
  format: string;
  clean_response: boolean;
  remove_hedging: boolean;
  remove_references: boolean;
  remove_disclaimers: boolean;
  ensure_html_structure: boolean;
  custom_hedging_patterns: string[];
  custom_reference_patterns: string[];
  custom_disclaimer_patterns: string[];
  html_tags_to_fix: string[];
};

export type NoInfoSettings = {
  no_info_title: string;
  no_info_message: string;
  include_suggestions: boolean;
  custom_suggestions: string[];
  no_info_html_format: boolean;
};

export function loadSettings() {
  // Default settings
  const defaultModelSettings: ModelSettings = {
    ollama_model: "llama3",
    ollama_embedding_model: "mxbai-embed-large",
    temperature: 0.7,
    top_p: 1.0,
    top_k_per_file: 5,
    similarity_threshold: 0.5,
    system_prompt: ""
  };

  const defaultFormatSettings: FormatSettings = {
    format: "html",
    clean_response: true,
    remove_hedging: true,
    remove_references: true,
    remove_disclaimers: true,
    ensure_html_structure: true,
    custom_hedging_patterns: [],
    custom_reference_patterns: [],
    custom_disclaimer_patterns: [],
    html_tags_to_fix: []
  };

  const defaultNoInfoSettings: NoInfoSettings = {
    no_info_title: "Information Not Found",
    no_info_message: "I couldn't find specific information about that in the selected document(s).",
    include_suggestions: true,
    custom_suggestions: [],
    no_info_html_format: true
  };

  // Load from localStorage if available
  let modelSettings = defaultModelSettings;
  let formatSettings = defaultFormatSettings;
  let noInfoSettings = defaultNoInfoSettings;

  if (typeof window !== 'undefined') {
    const savedModelSettings = localStorage.getItem("chatbot_model_settings");
    const savedFormatSettings = localStorage.getItem("chatbot_format_settings");
    const savedNoInfoSettings = localStorage.getItem("chatbot_noinfo_settings");
    
    if (savedModelSettings) {
      modelSettings = { ...defaultModelSettings, ...JSON.parse(savedModelSettings) };
    }
    
    if (savedFormatSettings) {
      formatSettings = { ...defaultFormatSettings, ...JSON.parse(savedFormatSettings) };
    }
    
    if (savedNoInfoSettings) {
      noInfoSettings = { ...defaultNoInfoSettings, ...JSON.parse(savedNoInfoSettings) };
    }
  }

  return {
    modelSettings,
    formatSettings,
    noInfoSettings
  };
}

export function applySettingsToRequest(baseRequest: any) {
  const { modelSettings, formatSettings, noInfoSettings } = loadSettings();
  
  return {
    ...baseRequest,
    ...modelSettings,
    ...formatSettings,
    ...noInfoSettings
  };
}