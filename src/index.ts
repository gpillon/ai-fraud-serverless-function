import express from 'express';
import { z } from 'zod';
import axios from 'axios';
import swaggerUi from 'swagger-ui-express';
import { OpenAPIV3 } from 'openapi-types';
import * as fs from 'fs';

const app = express();
const PORT = 8080;

// Load scaler values (instead of using pickle, we'll store the values directly)
const scalerData = JSON.parse(fs.readFileSync('src/scaler.json', 'utf-8'));

// Validation schema using zod
const QuerySchema = z.object({
  distance: z.string().transform(Number),
  ratio_to_median: z.string().transform(Number),
  pin: z.string().transform(Number).refine(val => val === 0 || val === 1),
  chip: z.string().transform(Number).refine(val => val === 0 || val === 1),
  online: z.string().transform(Number).refine(val => val === 0 || val === 1),
});

// OpenAPI documentation
const openApiSpec: OpenAPIV3.Document = {
  openapi: '3.0.0',
  info: {
    title: 'Fraud Detection API',
    version: '1.0.0',
    description: 'API for detecting fraudulent credit card transactions',
  },
  paths: {
    '/predict': {
      get: {
        summary: 'Predict fraud probability',
        parameters: [
          {
            name: 'distance',
            in: 'query',
            required: true,
            schema: { type: 'number' },
            description: 'Distance from last transaction location in km',
          },
          {
            name: 'ratio_to_median',
            in: 'query',
            required: true,
            schema: { type: 'number' },
            description: 'Ratio of transaction amount to median amount',
          },
          {
            name: 'pin',
            in: 'query',
            required: true,
            schema: { type: 'integer', enum: [0, 1] },
            description: 'PIN used (1) or not (0)',
          },
          {
            name: 'chip',
            in: 'query',
            required: true,
            schema: { type: 'integer', enum: [0, 1] },
            description: 'Chip used (1) or not (0)',
          },
          {
            name: 'online',
            in: 'query',
            required: true,
            schema: { type: 'integer', enum: [0, 1] },
            description: 'Online transaction (1) or not (0)',
          },
        ],
        responses: {
          '200': {
            description: 'Fraud prediction result',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    is_fraud: { type: 'boolean' },
                    fraud_probability: { type: 'number' },
                  },
                },
              },
            },
          },
        },
      },
    },
    '/health': {
      get: {
        summary: 'Health check',
        responses: {
          '200': {
            description: 'Service health status',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    status: { type: 'string' },
                  },
                },
              },
            },
          },
        },
      },
    },
  },
};

// Middleware to handle errors - ora con il parametro 'next' richiesto da Express
app.use((err: Error, _: express.Request, res: express.Response, _unused: express.NextFunction) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

// Swagger documentation route
app.use('/docs', swaggerUi.serve, swaggerUi.setup(openApiSpec));

// Function to normalize data using the scaler values
function normalizeData(data: number[]): number[] {
  return data.map((value, index) => {
    return (value - scalerData.mean[index]) / scalerData.scale[index];
  });
}

// Health check endpoint
app.get('/health', (_: express.Request, res: express.Response) => {
  res.json({ status: 'healthy' });
});

// Prediction endpoint
app.get('/predict', async (req: express.Request, res: express.Response, next: express.NextFunction) => {
  try {
    // Validate query parameters
    const validatedData = QuerySchema.parse(req.query);
    const rawData = [
      validatedData.distance,
      validatedData.ratio_to_median,
      validatedData.pin,
      validatedData.chip,
      validatedData.online,
    ];

    // Normalize the data
    const normalizedData = normalizeData(rawData);

    // Call the model service

    const agent = new https.Agent({  
        rejectUnauthorized: false
    });
    const response = await axios.post(process.env.FRAUD_MODEL_URL || '', {
      inputs: [{
        name: 'dense_input',
        shape: [1, 5],
        datatype: 'FP32',
        data: normalizedData,
      }],
    }, 
    { httpsAgent: agent });

    const fraudProbability = response.data.outputs[0].data[0];
    const threshold = Number(process.env.FRAUD_THRESHOLD || 0.95);

    res.json({
      is_fraud: fraudProbability > threshold,
      fraud_probability: fraudProbability,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({ error: 'Invalid input parameters', details: error.errors });
    } else {
      next(error); // Passa l'errore al middleware di gestione errori
    }
  }
});

// Root endpoint
app.get('/', (_: express.Request, res: express.Response) => {
  res.json({
    message: 'Welcome to Fraud Detection API. Visit /docs for Swagger documentation',
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
