# Multi-stage: build the bundle with Node, serve it with nginx.
# The runtime image carries no Node and no node_modules, just static files.

FROM node:20-alpine AS build

WORKDIR /app

# package files first so dependency install caches independently of source edits
COPY package.json package-lock.json ./

# Prefer `npm ci` for reproducibility, but fall back to `npm install` when the
# lockfile is out of sync. Vite 8 / rolldown and Tailwind 4 ship platform
# specific native binaries, so a lockfile generated on macOS can legitimately
# not satisfy `npm ci` inside a linux container. Failing the whole build over
# that is worse than a non-reproducible install for a local demo image.
RUN npm ci --no-audit --no-fund || \
    (echo "npm ci failed (lockfile drift); falling back to npm install" && \
     npm install --no-audit --no-fund)

COPY . .

# VITE_* vars are inlined at build time, not read at runtime. Left empty so API
# calls stay relative and nginx proxies them; the agent is addressed absolutely
# because the browser reaches it on a published host port, not over the compose
# network.
ARG VITE_API_URL=""
ARG VITE_AGENT_URL="http://localhost:8002"
ENV VITE_API_URL=$VITE_API_URL \
    VITE_AGENT_URL=$VITE_AGENT_URL

RUN npm run build


FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
