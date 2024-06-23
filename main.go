package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
	"os"
    "strconv"

	"github.com/valyala/fasthttp"
	"golang.org/x/sync/semaphore"
)

var (
    MODEL           = "claude-3-5-sonnet@20240620"
    PROJECT_ID      = getEnv("PROJECT_ID", "")
    CLIENT_ID       = getEnv("CLIENT_ID", "")
    CLIENT_SECRET   = getEnv("CLIENT_SECRET", "")
    REFRESH_TOKEN   = getEnv("REFRESH_TOKEN", "")
    API_KEY         = getEnv("API_KEY", "sk-pass")
    TOKEN_URL       = "https://www.googleapis.com/oauth2/v4/token"
    MAX_CONCURRENT  = getEnvAsInt64("MAX_CONCURRENT", 100)
)

// getEnv 从环境变量获取值，如果环境变量为空则返回默认值
func getEnv(key, defaultValue string) string {
    value := os.Getenv(key)
    if value == "" {
        return defaultValue
    }
    return value
}

// getEnvAsInt64 从环境变量获取int64值，如果环境变量为空或无法解析则返回默认值
func getEnvAsInt64(key string, defaultValue int64) int64 {
    strValue := os.Getenv(key)
    if strValue == "" {
        return defaultValue
    }
    intValue, err := strconv.ParseInt(strValue, 10, 64)
    if err != nil {
        return defaultValue
    }
    return intValue
}

var (
    accessToken     string
    tokenExpiry     int64
    tokenMutex      sync.RWMutex
    httpClient      = &http.Client{Timeout: 30 * time.Second}
    sem             = semaphore.NewWeighted(MAX_CONCURRENT)
)

type TokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

type RequestBody struct {
	Messages         []Message `json:"messages"`
	Stream           bool      `json:"stream"`
	MaxTokens        int       `json:"max_tokens,omitempty"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	TopK             int       `json:"top_k,omitempty"`
	AnthropicVersion string    `json:"anthropic_version,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func getAccessToken() (string, error) {
	tokenMutex.RLock()
	if time.Now().Unix() < tokenExpiry-60 {
		defer tokenMutex.RUnlock()
		return accessToken, nil
	}
	tokenMutex.RUnlock()

	tokenMutex.Lock()
	defer tokenMutex.Unlock()

	// Double check after acquiring the write lock
	if time.Now().Unix() < tokenExpiry-60 {
		return accessToken, nil
	}

	data := map[string]string{
		"client_id":     CLIENT_ID,
		"client_secret": CLIENT_SECRET,
		"refresh_token": REFRESH_TOKEN,
		"grant_type":    "refresh_token",
	}
	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", err
	}

	resp, err := httpClient.Post(TOKEN_URL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var tokenResp TokenResponse
	err = json.Unmarshal(body, &tokenResp)
	if err != nil {
		return "", err
	}

	accessToken = tokenResp.AccessToken
	tokenExpiry = time.Now().Unix() + int64(tokenResp.ExpiresIn)
	return accessToken, nil
}

func getLocation() string {
	if time.Now().Second() < 30 {
		return "europe-west1"
	}
	return "us-east5"
}

func constructApiUrl(location string) string {
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict", location, PROJECT_ID, location, MODEL)
}

func handleRequest(ctx *fasthttp.RequestCtx) {
	if string(ctx.Method()) == "OPTIONS" {
		handleOptions(ctx)
		return
	}

	if err := sem.Acquire(ctx, 1); err != nil {
		ctx.Error("Too many requests", fasthttp.StatusTooManyRequests)
		return
	}
	defer sem.Release(1)

	apiKey := string(ctx.Request.Header.Peek("x-api-key"))
	if apiKey != API_KEY {
		ctx.Error(`{"type":"error","error":{"type":"permission_error","message":"Your API key does not have permission to use the specified resource."}}`, fasthttp.StatusForbidden)
		return
	}

	accessToken, err := getAccessToken()
	if err != nil {
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	location := getLocation()
	apiUrl := constructApiUrl(location)

	var requestBody RequestBody
	if err := json.Unmarshal(ctx.PostBody(), &requestBody); err != nil {
		ctx.Error(err.Error(), fasthttp.StatusBadRequest)
		return
	}

	requestBody.AnthropicVersion = "vertex-2023-10-16"

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	req := fasthttp.AcquireRequest()
	defer fasthttp.ReleaseRequest(req)

	req.SetRequestURI(apiUrl)
	req.Header.SetMethod("POST")
	req.Header.SetContentType("application/json; charset=utf-8")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.SetBody(jsonData)

	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseResponse(resp)

	if err := fasthttp.Do(req, resp); err != nil {
		ctx.Error(err.Error(), fasthttp.StatusInternalServerError)
		return
	}

	ctx.SetStatusCode(resp.StatusCode())
	resp.Header.VisitAll(func(key, value []byte) {
		ctx.Response.Header.SetBytesV(string(key), value)
	})
	ctx.Response.Header.Set("Access-Control-Allow-Origin", "*")
	ctx.Response.Header.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	ctx.Response.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version, model")

	ctx.SetBody(resp.Body())
}

func handleOptions(ctx *fasthttp.RequestCtx) {
	ctx.Response.Header.Set("Access-Control-Allow-Origin", "*")
	ctx.Response.Header.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	ctx.Response.Header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version, model")
	ctx.SetStatusCode(fasthttp.StatusNoContent)
}

func main() {
	log.Println("Server starting on port 8080...")
	log.Fatal(fasthttp.ListenAndServe(":8080", handleRequest))
}