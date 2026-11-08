# Site AI chat — Cloudflare Worker

The "Ask about the paintings" chat on sveta-colours.com talks to OpenAI through
this Worker. The site is static (GitHub Pages), so the OpenAI key can't live in
`index.html`: anyone could read it from the page source and spend on it. The
Worker holds the key as a Cloudflare secret.

```
visitor's browser ──POST /chat {messages}──▶ Worker ──▶ OpenAI Chat Completions
   (sessionStorage keeps                   (key, system prompt,
    the whole conversation)                 live catalogue from the sheet CSV)
```

- **Conversation memory.** The widget keeps the conversation in the tab's
  `sessionStorage` (`sc-chat-v1`, last 40 messages) and sends it whole on every
  turn. The Worker passes the last 20 to the model. It survives a page reload
  and is gone when the tab closes. **Nothing is stored server-side.**
- **What the assistant knows.** Studio facts (About, FAQ, contact) are written
  into `chat-worker.js` (`STUDIO_FACTS`). The painting catalogue is read live
  from the published sheet CSV, so new paintings, prices and `sold` flags show
  up without a redeploy. The CSV is cached for 5 minutes.
- **Guards.** The Worker only answers requests from sveta-colours.com (and
  localhost:8000 for testing). It allows 10 messages per minute per IP, 1000
  characters per message, and 500 tokens per reply. OpenAI errors are never shown
  to visitors. **Also set a monthly budget in the OpenAI dashboard.** That's the
  only hard cap on spend.

## Deploy (one time)

Needs Node.js (`brew install node`) and a Cloudflare account (free plan is enough).

```bash
cd worker
npx wrangler login
npx wrangler deploy
echo "$OPENAI_API_KEY" | npx wrangler secret put OPENAI_API_KEY
```

`wrangler deploy` prints the URL, e.g.
`https://sveta-colours-chat.<account>.workers.dev`. Put it, with `/chat`, into
`CHAT_ENDPOINT_PROD` in `index.html`, then commit and push. Until that constant
is set, the chat button doesn't appear on the site at all.

Change the model with `OPENAI_MODEL` in `wrangler.toml` and redeploy.

## Test locally

Serve the repo (`scripts/dev.sh`), run `npx wrangler dev` in `worker/`
(port 8787), and open:

```
http://localhost:8000/?chat=http://localhost:8787/chat
```

The `?chat=` override only works on localhost.
