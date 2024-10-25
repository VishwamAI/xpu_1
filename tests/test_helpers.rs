use xpu_manager_rust::{
    xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, UserRole, Permission},
    XpuOptimizerError,
};
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize test environment
pub fn initialize_test_env(optimizer: &mut XpuOptimizer) -> Result<(), XpuOptimizerError> {
    INIT.call_once(|| {
        // Set up any global test configuration here
        if let Err(e) = optimizer.set_jwt_secret(b"test_secret_key_for_development_only".to_vec()) {
            eprintln!("Failed to set JWT secret: {}", e);
        }
    });
    Ok(())
}

/// Set up a test user with specified role
pub fn setup_test_user_with_role(optimizer: &mut XpuOptimizer, role: UserRole) -> Result<String, XpuOptimizerError> {
    initialize_test_env(optimizer)?;

    let username = match role {
        UserRole::Admin => "test_admin",
        UserRole::Manager => "test_manager",
        UserRole::User => "test_user",
    };

    // Add user with specified role
    optimizer.add_user(
        username.to_string(),
        "test_password".to_string(),
        role,
    ).map_err(|e| XpuOptimizerError::ConfigError(format!("Failed to add {} user: {}", username, e)))?;

    // Authenticate and get a valid token
    optimizer.authenticate_user(username, "test_password")
        .map_err(|e| XpuOptimizerError::AuthenticationError(format!("Failed to authenticate {} user: {}", username, e)))
}

/// Set up a test user with User role (backward compatibility)
pub fn setup_test_user(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    setup_test_user_with_role(optimizer, UserRole::User)
}

/// Set up an admin user for tests requiring elevated permissions
pub fn setup_admin_user(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    setup_test_user_with_role(optimizer, UserRole::Admin)
}

/// Set up a manager user for tests requiring management permissions
pub fn setup_manager_user(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    setup_test_user_with_role(optimizer, UserRole::Manager)
}

/// Validate a token and check permissions
pub fn validate_token_with_permission(
    optimizer: &XpuOptimizer,
    token: &str,
    required_permission: Permission,
) -> bool {
    match optimizer.get_user_from_token(token) {
        Ok(user) => optimizer.check_user_permission(&user.role, required_permission)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Create a test session with specified role
pub fn create_test_session(optimizer: &mut XpuOptimizer, role: UserRole) -> Result<String, XpuOptimizerError> {
    initialize_test_env(optimizer)?;

    let username = match role {
        UserRole::Admin => "session_admin",
        UserRole::Manager => "session_manager",
        UserRole::User => "session_user",
    };

    // Add user and create session
    optimizer.add_user(
        username.to_string(),
        "session_password".to_string(),
        role,
    ).map_err(|e| XpuOptimizerError::ConfigError(format!("Failed to add session user: {}", e)))?;

    optimizer.authenticate_user(username, "session_password")
        .map_err(|e| XpuOptimizerError::AuthenticationError(format!("Failed to authenticate session user: {}", e)))
}

/// Clean up test users and sessions
pub fn cleanup_test_data(optimizer: &mut XpuOptimizer) -> Result<(), XpuOptimizerError> {
    let test_users = vec![
        "test_admin", "test_manager", "test_user",
        "session_admin", "session_manager", "session_user"
    ];

    for username in test_users {
        if let Err(e) = optimizer.remove_user(username) {
            eprintln!("Warning: Failed to remove user {}: {}", username, e);
        }
    }
    Ok(())
}

/// Get a valid token for integration tests
pub fn get_integration_test_token(optimizer: &mut XpuOptimizer) -> Result<String, XpuOptimizerError> {
    initialize_test_env(optimizer)?;
    setup_test_user_with_role(optimizer, UserRole::Admin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_roles_and_permissions() -> Result<(), XpuOptimizerError> {
        let mut optimizer = XpuOptimizer::new(XpuOptimizerConfig::default())
            .map_err(|e| XpuOptimizerError::ConfigError(format!("Failed to create optimizer: {}", e)))?;
        initialize_test_env(&mut optimizer)?;

        // Test admin user
        let admin_token = setup_admin_user(&mut optimizer)?;
        assert!(validate_token_with_permission(&optimizer, &admin_token, Permission::ManageUsers));

        // Test manager user
        let manager_token = setup_manager_user(&mut optimizer)?;
        assert!(validate_token_with_permission(&optimizer, &manager_token, Permission::AddTask));
        assert!(!validate_token_with_permission(&optimizer, &manager_token, Permission::ManageUsers));

        // Test regular user
        let user_token = setup_test_user(&mut optimizer)?;
        assert!(validate_token_with_permission(&optimizer, &user_token, Permission::ViewTasks));
        assert!(!validate_token_with_permission(&optimizer, &user_token, Permission::AddSecureTask));

        cleanup_test_data(&mut optimizer)?;
        Ok(())
    }
}