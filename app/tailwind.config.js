/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./app/templates/**/*.html",
        "./app/static/js/**/*.js",
    ],
    theme: {
        extend: {
            fontFamily: {
                'inter': ['Inter', 'sans-serif'],
            },
            colors: {
                'primary': {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1d4ed8',
                },
                'secondary': {
                    500: '#f59e0b',
                    600: '#d97706',
                },
                'amber': {
                    50: '#fffbeb',
                    100: '#fef3c7',
                    500: '#f59e0b',
                    600: '#d97706',
                    700: '#b45309',
                }
            },
            animation: {
                'fade-in-up': 'fadeInUp 0.8s ease-out',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'spin-slow': 'spin 2s linear infinite',
            },
            keyframes: {
                fadeInUp: {
                    '0%': {
                        opacity: '0',
                        transform: 'translateY(30px)',
                    },
                    '100%': {
                        opacity: '1',
                        transform: 'translateY(0)',
                    },
                },
            },
        }
    },
    plugins: [],
}