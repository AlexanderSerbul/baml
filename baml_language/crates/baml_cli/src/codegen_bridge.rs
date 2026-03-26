//! Bridge from the compiler2 HIR/TIR to `baml_codegen_types::ObjectPool`.
//!
//! Walks the HIR item trees for user-defined files, resolves types via TIR,
//! and populates a codegen-ready `ObjectPool` suitable for language-specific
//! code generators (e.g. `baml_codegen_python`).

use anyhow::Result;
use baml_base::Name;
use baml_codegen_types::{self as cg, Namespace, ObjectPool};
use baml_compiler2_ast::DeclarativeMeta;
use baml_compiler2_hir::{self, file_package, package::PackageId};
use baml_compiler2_tir::{
    lower_type_expr,
    ty::{PrimitiveType, Ty as TirTy},
};
use baml_project::ProjectDatabase;

/// Build a codegen `ObjectPool` from the compiler database.
///
/// Walks all user-package source files, extracts classes/enums/type aliases/
/// declarative functions, resolves their types, and converts to codegen types.
pub fn build_object_pool(db: &ProjectDatabase) -> Result<ObjectPool> {
    let mut pool = ObjectPool::new();

    let user_pkg_id = PackageId::new(db, Name::new("user"));
    let user_pkg_items = baml_compiler2_ppir::package_items(db, user_pkg_id);

    for source_file in db.get_source_files() {
        let pkg_info = file_package::file_package(db, source_file);
        if pkg_info.package.as_str() != "user" {
            continue;
        }

        let item_tree = baml_compiler2_hir::file_item_tree(db, source_file);

        // Classes
        for (_id, class) in &item_tree.classes {
            if !class.generic_params.is_empty() {
                continue;
            }
            let cg_name = cg::Name {
                name: class.name.clone(),
                namespace: Namespace::Types,
            };
            let properties = class
                .fields
                .iter()
                .filter_map(|field| {
                    let ty = resolve_type_expr(
                        db,
                        field.type_expr.as_ref(),
                        user_pkg_items,
                        &pkg_info.namespace_path,
                        &[],
                    )?;
                    Some(cg::ClassProperty {
                        name: field.name.clone(),
                        docstring: None,
                        ty,
                    })
                })
                .collect();
            pool.insert(
                cg_name.clone(),
                cg::Object::Class(cg::Class {
                    name: cg_name,
                    docstring: None,
                    properties,
                }),
            );
        }

        // Enums
        for (_id, enum_def) in &item_tree.enums {
            let cg_name = cg::Name {
                name: enum_def.name.clone(),
                namespace: Namespace::Types,
            };
            let variants = enum_def
                .variants
                .iter()
                .map(|v| cg::EnumVariant {
                    name: v.name.clone(),
                    docstring: None,
                    value: v.name.to_string(),
                })
                .collect();
            pool.insert(
                cg_name.clone(),
                cg::Object::Enum(cg::Enum {
                    name: cg_name,
                    docstring: None,
                    variants,
                }),
            );
        }

        // Type aliases
        for (_id, alias) in &item_tree.type_aliases {
            if let Some(resolved) = resolve_type_expr(
                db,
                alias.type_expr.as_ref(),
                user_pkg_items,
                &pkg_info.namespace_path,
                &[],
            ) {
                let cg_name = cg::Name {
                    name: alias.name.clone(),
                    namespace: Namespace::Types,
                };
                pool.insert(
                    cg_name.clone(),
                    cg::Object::TypeAlias(cg::TypeAlias {
                        name: cg_name,
                        resolves_to: resolved,
                    }),
                );
            }
        }

        // Functions (only declarative LLM functions)
        for (_id, func) in &item_tree.functions {
            if !matches!(&func.declarative_meta, Some(DeclarativeMeta::Llm(_))) {
                continue;
            }
            if !func.generic_params.is_empty() {
                continue;
            }

            let arguments: Vec<cg::FunctionArgument> = func
                .params
                .iter()
                .filter_map(|param| {
                    let ty = resolve_type_expr(
                        db,
                        param.type_expr.as_ref(),
                        user_pkg_items,
                        &pkg_info.namespace_path,
                        &[],
                    )?;
                    Some(cg::FunctionArgument {
                        name: param.name.clone(),
                        docstring: None,
                        ty,
                    })
                })
                .collect();

            let return_type = resolve_type_expr(
                db,
                func.return_type.as_ref(),
                user_pkg_items,
                &pkg_info.namespace_path,
                &[],
            )
            .unwrap_or(cg::Ty::Unit);

            let cg_name = cg::Name {
                name: func.name.clone(),
                namespace: Namespace::Types,
            };
            pool.insert(
                cg_name,
                cg::Object::Function(cg::Function {
                    name: func.name.clone(),
                    docstring: None,
                    arguments,
                    return_type,
                    stream_return_type: None, // TODO: streaming support
                    watchers: Vec::new(),
                }),
            );
        }
    }

    Ok(pool)
}

/// Resolve an optional `SpannedTypeExpr` to a codegen `Ty`.
///
/// Returns `None` if the type expression is missing.
fn resolve_type_expr(
    db: &ProjectDatabase,
    spanned: Option<&baml_compiler2_ast::SpannedTypeExpr>,
    package_items: &baml_compiler2_hir::package::PackageItems<'_>,
    ns_context: &[Name],
    generic_params: &[Name],
) -> Option<cg::Ty> {
    let spanned = spanned?;
    let mut diagnostics = Vec::new();
    let tir_ty = lower_type_expr::lower_type_expr_in_ns(
        db,
        &spanned.expr,
        package_items,
        ns_context,
        generic_params,
        &mut diagnostics,
    );
    Some(convert_tir_to_codegen_ty(&tir_ty))
}

/// Convert a TIR `Ty` to a `baml_codegen_types::Ty`.
fn convert_tir_to_codegen_ty(ty: &TirTy) -> cg::Ty {
    match ty {
        // Primitives
        TirTy::Primitive(PrimitiveType::Int) => cg::Ty::Int,
        TirTy::Primitive(PrimitiveType::Float) => cg::Ty::Float,
        TirTy::Primitive(PrimitiveType::String) => cg::Ty::String,
        TirTy::Primitive(PrimitiveType::Bool) => cg::Ty::Bool,
        TirTy::Primitive(PrimitiveType::Null) => cg::Ty::Null,
        TirTy::Primitive(PrimitiveType::Image) => cg::Ty::Media(baml_base::MediaKind::Image),
        TirTy::Primitive(PrimitiveType::Audio) => cg::Ty::Media(baml_base::MediaKind::Audio),
        TirTy::Primitive(PrimitiveType::Video) => cg::Ty::Media(baml_base::MediaKind::Video),
        TirTy::Primitive(PrimitiveType::Pdf) => cg::Ty::Media(baml_base::MediaKind::Pdf),

        // Named types
        TirTy::Class(qtn) => cg::Ty::Class(cg::Name {
            name: qtn.name().clone(),
            namespace: Namespace::Types,
        }),
        TirTy::Enum(qtn) => cg::Ty::Enum(cg::Name {
            name: qtn.name().clone(),
            namespace: Namespace::Types,
        }),
        TirTy::EnumVariant(qtn, _variant) => cg::Ty::Enum(cg::Name {
            name: qtn.name().clone(),
            namespace: Namespace::Types,
        }),
        // Type aliases: keep as the alias name (the ObjectPool has the alias entry)
        TirTy::TypeAlias(qtn) => cg::Ty::Class(cg::Name {
            name: qtn.name().clone(),
            namespace: Namespace::Types,
        }),

        // Containers
        TirTy::List(inner) | TirTy::EvolvingList(inner) => {
            cg::Ty::List(Box::new(convert_tir_to_codegen_ty(inner)))
        }
        TirTy::Map(k, v) | TirTy::EvolvingMap(k, v) => cg::Ty::Map {
            key: Box::new(convert_tir_to_codegen_ty(k)),
            value: Box::new(convert_tir_to_codegen_ty(v)),
        },
        TirTy::Union(members) => {
            cg::Ty::Union(members.iter().map(convert_tir_to_codegen_ty).collect())
        }
        TirTy::Optional(inner) => {
            cg::Ty::Optional(Box::new(convert_tir_to_codegen_ty(inner)))
        }
        TirTy::Literal(lit, _freshness) => cg::Ty::Literal(lit.clone()),

        // Bottom / sentinel / error recovery
        TirTy::Void
        | TirTy::Never
        | TirTy::Unknown
        | TirTy::Error
        | TirTy::TypeVar(_)
        | TirTy::BuiltinUnknown
        | TirTy::RustType
        | TirTy::Type => cg::Ty::Unit,

        // Function types don't map to codegen types
        TirTy::Function { .. } => cg::Ty::Unit,
    }
}
