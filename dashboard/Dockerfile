FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source files
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Run Vite with network exposure so host can access it
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
