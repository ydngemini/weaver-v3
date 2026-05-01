#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const VENV_PATH = path.join(process.cwd(), 'weaver_env');

function setupEnvironment() {
    console.log('Checking Weaver V3 dependencies...');

    if (!fs.existsSync(VENV_PATH)) {
        console.log('Creating virtual environment...');
        execSync(`python3 -m venv ${VENV_PATH}`, { stdio: 'inherit' });
    }

    console.log('Syncing Python requirements...');
    const pipPath = path.join(VENV_PATH, 'bin', 'pip');
    const reqPath = path.join(__dirname, 'requirements.txt');

    try {
        execSync(`${pipPath} install --upgrade pip`, { stdio: 'inherit' });
        execSync(`${pipPath} install -r ${reqPath}`, { stdio: 'inherit' });
        console.log('Environment is up to date.');
    } catch (error) {
        console.error('Failed to update dependencies:', error.message);
        process.exit(1);
    }
}

setupEnvironment();
