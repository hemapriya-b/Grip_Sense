// Profile logic: validation for caretaker signup and the shared patient profile.
// Each validator returns { fieldName: "error message" }; an empty object means valid.

export const GENDERS = ["Male", "Female", "Other", "Prefer not to say"];

const USERNAME_MAX_LENGTH = 50;
const PATIENT_NAME_MAX_LENGTH = 50;
const CONDITION_MAX_LENGTH = 300;
const PASSWORD_MIN_LENGTH = 8;
const AGE_MIN = 0;
const AGE_MAX = 120;

const CONTROL_CHARS = /\p{Cc}/u;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function checkText(value, label, maxLength) {
    if (!value) return `Please enter the ${label}.`;
    if (value.length > maxLength) return `${label[0].toUpperCase()}${label.slice(1)} must be ${maxLength} characters or fewer.`;
    if (CONTROL_CHARS.test(value)) return `${label[0].toUpperCase()}${label.slice(1)} contains invalid characters.`;
    return null;
}

// Returns an error message, or null when the name is acceptable.
export function validatePatientName(name) {
    return checkText(name, "patient name", PATIENT_NAME_MAX_LENGTH);
}

export function validateUsername(username) {
    return checkText(username, "username", USERNAME_MAX_LENGTH);
}

export function validateSignup({ username, email, password, confirmPassword }) {
    const errors = {};
    const usernameError = validateUsername(username);
    if (usernameError) errors.username = usernameError;

    if (!email) errors.email = "Please enter your email address.";
    else if (!EMAIL_PATTERN.test(email)) errors.email = "Please enter a valid email address.";

    if (!password) errors.password = "Please enter a password.";
    else if (password.length < PASSWORD_MIN_LENGTH) {
        errors.password = `Password must be at least ${PASSWORD_MIN_LENGTH} characters.`;
    }

    if (!confirmPassword) errors.confirmPassword = "Please confirm your password.";
    else if (password !== confirmPassword) errors.confirmPassword = "Passwords do not match.";

    return errors;
}

// Accepts form strings or stored values; age must be a whole number 0–120.
export function validatePatientProfile({ name, age, gender, condition }) {
    const errors = {};
    const nameError = validatePatientName(name);
    if (nameError) errors.name = nameError;

    const ageText = String(age ?? "").trim();
    if (!ageText) errors.age = "Please enter the patient's age.";
    else if (!/^\d+$/.test(ageText) || Number(ageText) < AGE_MIN || Number(ageText) > AGE_MAX) {
        errors.age = `Age must be a whole number from ${AGE_MIN} to ${AGE_MAX}.`;
    }

    if (!GENDERS.includes(gender)) errors.gender = "Please select a gender.";

    const conditionError = checkText(condition, "condition", CONDITION_MAX_LENGTH);
    if (conditionError) errors.condition = conditionError;

    return errors;
}

// True when the stored /patient record has every required field.
export function isPatientComplete(patient) {
    if (!patient || typeof patient !== "object") return false;
    const typedCorrectly = typeof patient.name === "string"
        && Number.isInteger(patient.age)
        && typeof patient.condition === "string";
    return typedCorrectly && Object.keys(validatePatientProfile(patient)).length === 0;
}
