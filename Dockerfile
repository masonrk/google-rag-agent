FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY agent.ts tsconfig.json ./

ENV NODE_ENV=production
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "NODE_NO_WARNINGS=1 npx adk api_server agent.ts --host 0.0.0.0 --port ${PORT:-8080} --log_level error"]
