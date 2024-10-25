use crate::xpu_optimization::{XpuOptimizer, XpuOptimizerConfig, UserRole, Permission};
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize test environment
pub fn initialize_test_env(optimizer: &mut XpuOptimizer) {
    INIT.call_once(|| {
        // Set up any global test configuration here
        optimizer.set_jwt_secret(b"test_secret_key_for_development_only".to_vec());
    });
}

/// Set up a test user with specified role
pub fn setup_test_user_with_role(optimizer: &mut XpuOptimizer, role: UserRole) -> String {
    initialize_test_env(optimizer);

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
    ).expect(&format!("Failed to add {} user", username));

    // Authenticate and get a valid token
    optimizer.authenticate_user(username, "test_password")
        .expect(&format!("Failed to authenticate {} user", username))
}

/// Set up a test user with User role (backward compatibility)
pub fn setup_test_user(optimizer: &mut XpuOptimizer) -> String {
    setup_test_user_with_role(optimizer, UserRole::User)
}

/// Set up an admin user for tests requiring elevated permissions
pub fn setup_admin_user(optimizer: &mut XpuOptimizer) -> String {
    setup_test_user_with_role(optimizer, UserRole::Admin)
}

/// Set up a manager user for tests requiring management permissions
pub fn setup_manager_user(optimizer: &mut XpuOptimizer) -> String {
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
pub fn create_test_session(optimizer: &mut XpuOptimizer, role: UserRole) -> String {
    initialize_test_env(optimizer);

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
    ).expect("Failed to add session user");

    optimizer.authenticate_user(username, "session_password")
        .expect("Failed to authenticate session user")
}

/// Clean up test users and sessions
pub fn cleanup_test_data(optimizer: &mut XpuOptimizer) {
    let test_users = vec![
        "test_admin", "test_manager", "test_user",
        "session_admin", "session_manager", "session_user"
    ];

    for username in test_users {
        let _ = optimizer.remove_user(username);
    }
}

/// Get a valid token for integration tests
pub fn get_integration_test_token(optimizer: &mut XpuOptimizer) -> String {
    initialize_test_env(optimizer);
    setup_test_user_with_role(optimizer, UserRole::Admin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_roles_and_permissions() {
        let mut optimizer = XpuOptimizer::new(XpuOptimizerConfig::default());
        initialize_test_env(&mut optimizer);

        // Test admin user
        let admin_token = setup_admin_user(&mut optimizer);
        assert!(validate_token_with_permission(&optimizer, &admin_token, Permission::ManageUsers));

        // Test manager user
        let manager_token = setup_manager_user(&mut optimizer);
        assert!(validate_token_with_permission(&optimizer, &manager_token, Permission::AddTask));
        assert!(!validate_token_with_permission(&optimizer, &manager_token, Permission::ManageUsers));

        // Test regular user
        let user_token = setup_test_user(&mut optimizer);
        assert!(validate_token_with_permission(&optimizer, &user_token, Permission::ViewTasks));
        assert!(!validate_token_with_permission(&optimizer, &user_token, Permission::AddSecureTask));

        cleanup_test_data(&mut optimizer);
    }
}
